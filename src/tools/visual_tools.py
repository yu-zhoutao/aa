import os
import cv2
import uuid
import asyncio
from typing import Dict, List, Any
from ..config import Config
from .base import BaseTool
from ..engines.yolo_engine import YoloEngine
from ..engines.face_engine import FaceEngine
from ..engines.ocr_engine import OcrEngine
from ..engines.minio_engine import MinioEngine
from ..engines.llm_engine import LLMEngine
from ..utils.image_utils import ImageUtils
from ..utils.evidence_utils import EvidenceUtils
from ..utils.json_utils import JSONUtils
from ..prompts import PromptTemplates

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
        frames = _normalize_frames_input(frames, image_path)
        if not frames:
            return {"error": "必须提供 frames 或 image_path"}

        previews = []
        frame_previews = []

        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            try:
                minio_url = MinioEngine.upload_file(path)
                previews.append(minio_url)
                frame_previews.append({"index": item["index"], "preview": minio_url})
            except Exception as e:
                print(f"⚠️ 预览图上传失败，回退到 Base64: {e}")
                img = cv2.imread(path)
                if img is not None:
                    b64 = ImageUtils.encode_to_base64(img)
                    b64_str = f"data:image/jpeg;base64,{b64}"
                    previews.append(b64_str)
                    frame_previews.append({"index": item["index"], "preview": b64_str})

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
                try: item["minio_url"] = MinioEngine.upload_file(item["path"])
                except: pass

        if not frames: return {"error": "无有效图片数据"}

        persons = []
        detected_persons = []
        visual_risks = []
        evidence_bboxes = [] # 收集证据框

        for item in frames:
            minio_url = item.get("minio_url")
            if not minio_url: continue

            try:
                results = await asyncio.to_thread(FaceEngine.identify_face, minio_url)
                if results:
                    for p in results:
                        p_name = p.get("name", "未知")
                        p_tag = p.get("tag", "")
                        p_bbox = p.get("bbox", []) # [x1, y1, x2, y2]
                        
                        p_info = f"{p_name} ({p_tag})"
                        detected_persons.append(p_info)
                        
                        if "黑名单" in p_tag or "敏感" in p_tag or "落马" in p_tag:
                             visual_risks.append(f"发现敏感人物: {p_info}")
                        
                        if p_bbox:
                             evidence_bboxes.append({
                                 "frame_index": item["index"],
                                 "bbox": p_bbox,
                                 "label": p_name,
                                 "color": (0, 0, 255) # 红色
                             })
                        
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
            "visual_risks": list(dict.fromkeys(visual_risks)),
            "evidence_bboxes": evidence_bboxes # 返回框数据
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

class OcrRiskJudgeTool(BaseTool):
    def __init__(self):
        super().__init__("ocr_risk_judge", "OCR 敏感文本判定与标注")

    async def run(self, ocr_results: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        if not ocr_results: return {"error": "无 OCR 数据"}
        
        risks = []
        evidence_bboxes = []

        for item in ocr_results:
            texts_data = item.get("items", [])
            if not texts_data: continue

            full_text = " ".join([t.get("text", "") for t in texts_data])
            if len(full_text) > 4000: full_text = full_text[:4000] + "..."
            
            text_map = {t.get("id", i): t.get("text", "") for i, t in enumerate(texts_data[:100])}
            
            prompt = PromptTemplates.ocr_judge_prompt(full_text, text_map)
            msgs = [{"role": "user", "content": prompt}]
            try:
                client = LLMEngine.get_client()
                response_text = await client.ainvoke(prompt, user_prompt="")
                res = JSONUtils.safe_json_loads(response_text)
                
                if res and isinstance(res, dict) and res.get("id"):
                    hit_ids = res["id"]
                    if not isinstance(hit_ids, list): hit_ids = [hit_ids]
                    hit_ids_str = [str(x) for x in hit_ids]
                    
                    risks.append(f"发现敏感文字: {res.get('reason', '未知原因')}")
                    
                    for t in texts_data:
                        t_id = t.get("id")
                        if t_id is not None and str(t_id) in hit_ids_str:
                            if "box" in t:
                                pts = t["box"]
                                if isinstance(pts, (list, tuple)) and len(pts) >= 4:
                                    try:
                                        xs = [p[0] for p in pts]
                                        ys = [p[1] for p in pts]
                                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                                        evidence_bboxes.append({
                                            "frame_index": item.get("index", 0),
                                            "bbox": bbox,
                                            "label": "敏感文字",
                                            "color": (0, 255, 0) # 🟢 绿色 (BGR: Green)
                                        })
                                    except (IndexError, TypeError):
                                        pass

            except Exception as e:
                print(f"OCR 判定异常: {e}")

        return {
            "status": "success",
            "ocr_risks": risks,
            "evidence_bboxes": evidence_bboxes 
        }

class BehaviorJudgeTool(BaseTool):
    def __init__(self):
        super().__init__("behavior_judge", "敏感行为/标识判定 (LLM)")

    async def run(self, frames: List[Dict[str, Any]] = None, bboxes: List[List[int]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        if not frames: return {"error": "无帧数据"}

        visual_risks = []
        violations = []
        evidence_bboxes = []

        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            img = cv2.imread(path)
            if img is None: continue

            # 如果没有传入 bboxes，尝试自动检测
            frame_bboxes = item.get("bboxes") or bboxes
            if not frame_bboxes:
                # 自动调用 YOLO 获取候选框
                raw_dets = YoloEngine.detect(img, conf=0.3)
                frame_bboxes = ImageUtils.merge_overlapping_boxes(raw_dets, img.shape)
            
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

            prompt = PromptTemplates.get_image_prompt("违规行为、敏感标识、阴暗内容、同性低俗、政治旗帜、国民党党旗、台独、台湾旗帜、丑化嘲讽领导人，歧视中国人")
            msgs = LLMEngine.build_visual_message(prompt, slices_b64)
            
            try:
                res = await LLMEngine.get_json_response(msgs)
            except:
                res = {}

            if res and res.get("image"):
                hit_ids = res["image"]
                hit_bboxes = []
                for idx in hit_ids:
                    if 0 < idx <= len(valid_bboxes):
                        hit_bboxes.append(valid_bboxes[idx-1])
                        visual_risks.append(f"发现敏感内容 (帧 {item['index']}, ID {idx})")
                
                if hit_bboxes:
                    for b in hit_bboxes:
                        if isinstance(b, dict) and "bbox" in b: bb = b["bbox"]
                        else: bb = b
                        evidence_bboxes.append({
                            "frame_index": item["index"],
                            "bbox": bb,
                            "label": res.get("reason", "违规内容")[:10],
                            "color": (0, 0, 255) # 红色
                        })

                    violations.append({
                        "index": item["index"],
                        "bboxes": hit_bboxes,
                        "reason": res.get("reason", "")
                    })

        return {
            "status": "success", 
            "visual_risks": visual_risks, 
            "violations": violations,
            "evidence_bboxes": evidence_bboxes
        }

class ImageAnnotateTool(BaseTool):
    def __init__(self): super().__init__("image_annotate", "（已停用，改用EvidenceUtils）")
    async def run(self, **kwargs): return {"status": "deprecated"}
