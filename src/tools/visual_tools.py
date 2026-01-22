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

class BehaviorJudgeTool(BaseTool):
    def __init__(self):
        super().__init__("behavior_judge", "敏感行为/标识判定 (LLM)")

    async def run(self, frames: List[Dict[str, Any]] = None, bboxes: List[List[int]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        if not frames: return {"error": "无帧数据"}

        visual_risks = []
        violations = []

        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            
            img = cv2.imread(path)
            if img is None: continue

            # 优先使用帧内特定的 bbox，否则使用全局传入的 bbox
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
                    violations.append({
                        "index": item["index"],
                        "bboxes": hit_bboxes,
                        "reason": res.get("reason", "")
                    })

        return {"status": "success", "visual_risks": visual_risks, "violations": violations}
