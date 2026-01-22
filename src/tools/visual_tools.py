import os
import cv2
import uuid
import asyncio
from typing import Dict, List, Any
from ..utils.config import Config
from .base import BaseTool
from ..engines.yolo_engine import YoloEngine
from ..engines.face_engine import FaceEngine
from ..engines.ocr_engine import OcrEngine
from ..engines.minio_engine import MinioEngine
from ..engines.llm_engine import LLMEngine
from ..utils.image_utils import ImageUtils
from ..utils.prompts import PromptTemplates

def _ensure_temp_dir():
    config = Config()
    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir)

def _normalize_frames_input(frames: Any = None, image_path: str = "") -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if frames:
        for item in frames:
            if isinstance(item, str):
                items.append({"path": item})
            elif isinstance(item, dict):
                items.append(dict(item))
    elif image_path:
        items.append({"path": image_path, "index": 0})

    for i, item in enumerate(items):
        if "index" not in item:
            item["index"] = i
    return items

class FrameExtractTool(BaseTool):
    def __init__(self):
        super().__init__("frame_extract", "从视频中提取关键帧或读取图片")

    async def run(self, file_path: str, sample_count: int = 8, **kwargs) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}

        _ensure_temp_dir()
        config = Config()
        
        frames_data = ImageUtils.extract_frames(file_path, sample_count=sample_count)
        if not frames_data:
            return {"error": "无法提取图像帧"}

        saved_frames = []
        for item in frames_data:
            temp_filename = f"frame_{item['index']}_{uuid.uuid4().hex}.jpg"
            temp_filepath = os.path.join(config.temp_dir, temp_filename)
            cv2.imwrite(temp_filepath, item["img"])
            saved_frames.append({"index": item["index"], "path": temp_filepath})

        return {"status": "success", "frames": saved_frames, "sample_count": sample_count}

class FrameUploadTool(BaseTool):
    def __init__(self):
        super().__init__("frame_upload", "上传帧到 MinIO")

    async def run(self, frames: List[Dict[str, Any]] = None, image_path: str = "", **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames, image_path)
        if not frames:
            return {"error": "必须提供 frames 或 image_path"}

        for item in frames:
            if item.get("minio_url"): continue
            path = item.get("path")
            if not path or not os.path.exists(path):
                item["minio_url"] = None
                continue
            try:
                item["minio_url"] = MinioEngine.upload_file(path)
            except Exception as e:
                print(f"⚠️ 上传失败: {e}")
                item["minio_url"] = None

        return {"status": "success", "frames": frames}

class PreviewUploadTool(BaseTool):
    def __init__(self):
        super().__init__("preview_upload", "生成并上传预览图")

    async def run(self, frames: List[Dict[str, Any]] = None, image_path: str = "", **kwargs) -> Dict[str, Any]:
        """上传图片并返回 preview_images 供前端展示"""
        frames = _normalize_frames_input(frames, image_path)
        if not frames:
            return {"error": "必须提供 frames 或 image_path"}

        previews = []
        frame_previews = []

        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path):
                continue
            try:
                minio_url = MinioEngine.upload_file(path)
                previews.append(minio_url)
                frame_previews.append({"index": item["index"], "preview": minio_url})
            except Exception as e:
                print(f"⚠️ 预览上传失败: {e}")
                img = cv2.imread(path)
                b64 = ImageUtils.encode_to_base64(img) if img is not None else ""
                previews.append(f"data:image/jpeg;base64,{b64}")

        return {"status": "success", "preview_images": previews, "frames": frame_previews}


class FaceIdentifyTool(BaseTool):
    def __init__(self):
        super().__init__("face_identify", "人脸识别 (黑名单/人物)")

    async def run(self, frames: List[Dict[str, Any]] = None, image_url: str = "", **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames, "")
        if image_url:
            frames.append({"index": 0, "minio_url": image_url})
        
        for item in frames:
            if not item.get("minio_url") and item.get("path"):
                try:
                    item["minio_url"] = MinioEngine.upload_file(item["path"])
                except:
                    pass

        if not frames:
            return {"error": "无有效图片数据"}

        persons = []
        detected_persons = []
        visual_risks = []

        for item in frames:
            minio_url = item.get("minio_url")
            if not minio_url: continue

            try:
                results = await asyncio.to_thread(FaceEngine.identify_face, minio_url)
                if results:
                    for p in results:
                        p_name = p.get("name", "未知")
                        p_tag = p.get("tag", "")
                        p_info = f"{p_name} ({p_tag})"
                        detected_persons.append(p_info)
                        if "黑名单" in p_tag or "敏感" in p_tag:
                             visual_risks.append(f"发现敏感人物: {p_info}")
                        
                        persons.append({
                            "index": item["index"],
                            "name": p_name,
                            "tag": p_tag,
                            "similarity": p.get("similarity", 0)
                        })
            except Exception as e:
                print(f"⚠️ 人脸识别异常: {e}")

        return {
            "status": "success", 
            "persons": persons, 
            "detected_persons": list(dict.fromkeys(detected_persons)),
            "visual_risks": list(dict.fromkeys(visual_risks))
        }


class YoloDetectTool(BaseTool):
    def __init__(self):
        super().__init__("yolo_detect", "目标检测 (YOLO)")

    async def run(self, frames: List[Dict[str, Any]] = None, image_path: str = "", conf: float = 0.3, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames, image_path)
        detections = []
        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            
            img = cv2.imread(path)
            if img is None: continue

            raw_detections = YoloEngine.detect(img, conf=conf)
            merged = ImageUtils.merge_overlapping_boxes(raw_detections, img.shape)
            detections.append({"index": item["index"], "path": path, "bboxes": merged})

        return {"status": "success", "detections": detections}

class OcrDetectTool(BaseTool):
    def __init__(self):
        super().__init__("ocr_detect", "OCR 文字检测")

    async def run(self, frames: List[Dict[str, Any]] = None, image_path: str = "", **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames, image_path)
        ocr_results = []
        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            
            img = cv2.imread(path)
            if img is None: continue
            
            results = OcrEngine.detect_text(img)
            ocr_results.append({"index": item["index"], "path": path, "items": results})

        return {"status": "success", "ocr_results": ocr_results}

class ImageAnnotateTool(BaseTool):
    def __init__(self):
        super().__init__("image_annotate", "图片标注")

    async def run(self, frames: List[Dict[str, Any]] = None, bboxes: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        if not frames: return {"error": "无帧数据"}
        _ensure_temp_dir()
        config = Config()
        
        annotated = []
        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            
            img = cv2.imread(path)
            if img is None: continue

            # 这里的 bboxes 结构可能需要适配
            # 假设 bboxes 是一个列表，里面是 {'bbox': [x1, y1, x2, y2]}
            # 但实际上 BehaviorJudgeTool 产生的是 violations 列表
            frame_violations = item.get("violations", [])
            
            # 如果传入了全局 bboxes
            if bboxes:
                frame_violations.extend(bboxes)

            if frame_violations:
                to_draw = []
                for v in frame_violations:
                    if isinstance(v, dict) and "bbox" in v:
                        to_draw.append(v)
                    elif isinstance(v, list) and len(v) == 4:
                        to_draw.append({"bbox": v})

                if to_draw:
                    img = ImageUtils.draw_detections(img, to_draw, color=(0, 0, 255), thickness=3)

            temp_filename = f"annotated_{item['index']}_{uuid.uuid4().hex}.jpg"
            temp_filepath = os.path.join(config.temp_dir, temp_filename)
            cv2.imwrite(temp_filepath, img)
            annotated.append({"index": item["index"], "path": temp_filepath})
            
        return {"status": "success", "annotated_frames": annotated}


class BehaviorJudgeTool(BaseTool):
    def __init__(self):
        super().__init__("behavior_judge", "敏感行为/标识判定 (LLM)")

    async def run(self, frames: List[Dict[str, Any]] = None, bboxes: List[List[int]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        if not frames: return {"error": "无帧数据"}

        visual_risks = []
        violations = []
        annotated_evidence = [] # 存储标注后的图片路径

        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            
            img = cv2.imread(path)
            if img is None: continue

            frame_bboxes = item.get("bboxes") or bboxes or []
            if not frame_bboxes: continue

            slices_b64 = []
            valid_bboxes = []
            for det in frame_bboxes:
                bbox = det.get('bbox') if isinstance(det, dict) else det
                if bbox:
                    crop = ImageUtils.get_single_object_crop(img, bbox)
                    slices_b64.append(ImageUtils.encode_to_base64(crop))
                    valid_bboxes.append(det)

            if not slices_b64: continue

            prompt = PromptTemplates.get_image_prompt("违规行为、敏感标识、阴暗内容、同性低俗、政治旗帜")
            msgs = LLMEngine.build_visual_message(prompt, slices_b64)
            res = await LLMEngine.get_json_response(msgs)

            if res and res.get("image"):
                hit_ids = res["image"]
                hit_bboxes = []
                for idx in hit_ids:
                    if 0 < idx <= len(valid_bboxes):
                        hit_bboxes.append(valid_bboxes[idx-1])
                        visual_risks.append(f"发现敏感内容 (帧 {item['index']}, ID {idx})")
                
                if hit_bboxes:
                    # 发现违规，立即调用标注工具
                    annotator = ImageAnnotateTool()
                    # 构造符合格式的 bboxes (如果 valid_bboxes 是纯坐标，需要包一层)
                    draw_bboxes = []
                    for b in hit_bboxes:
                        if isinstance(b, list): draw_bboxes.append({"bbox": b})
                        else: draw_bboxes.append(b)
                        
                    anno_res = await annotator.run([{"path": path, "index": item["index"]}], bboxes=draw_bboxes)
                    if anno_res.get("annotated_frames"):
                        evidence_path = anno_res["annotated_frames"][0]["path"]
                        annotated_evidence.append(evidence_path)
                        print(f"📸 生成违规证据图: {evidence_path}")

                    violations.append({
                        "index": item["index"],
                        "bboxes": hit_bboxes,
                        "reason": res.get("reason", ""),
                        "evidence_path": evidence_path if anno_res.get("annotated_frames") else ""
                    })

        return {
            "status": "success", 
            "visual_risks": visual_risks, 
            "violations": violations,
            "evidence_images": annotated_evidence # 返回给外部
        }
