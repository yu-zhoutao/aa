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
            if isinstance(item, str): items.append({"path": item})
            elif isinstance(item, dict): items.append(dict(item))
    elif image_path:
        items.append({"path": image_path, "index": 0})

    for i, item in enumerate(items):
        if "index" not in item: item["index"] = i
    return items

class FrameExtractTool(BaseTool):
    def __init__(self):
        super().__init__("frame_extract", "提取关键帧")

    async def run(self, file_path: str, sample_count: int = 8, **kwargs) -> Dict[str, Any]:
        if not os.path.exists(file_path): return {"error": f"文件不存在: {file_path}"}
        _ensure_temp_dir()
        config = Config()
        frames_data = ImageUtils.extract_frames(file_path, sample_count=sample_count)
        if not frames_data: return {"error": "无法提取图像帧"}
        saved_frames = []
        for item in frames_data:
            temp_filepath = os.path.join(config.temp_dir, f"frame_{item['index']}_{uuid.uuid4().hex[:6]}.jpg")
            cv2.imwrite(temp_filepath, item["img"])
            saved_frames.append({"index": item["index"], "path": temp_filepath})
        return {"status": "success", "frames": saved_frames}

class FrameUploadTool(BaseTool):
    def __init__(self):
        super().__init__("frame_upload", "上传帧到 MinIO")

    async def run(self, frames: List[Dict[str, Any]] = None, image_path: str = "", **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames, image_path)
        for item in frames:
            if item.get("minio_url"): continue
            path = item.get("path")
            if path and os.path.exists(path):
                try: item["minio_url"] = MinioEngine.upload_file(path)
                except: pass
        return {"status": "success", "frames": frames}

class PreviewUploadTool(BaseTool):
    def __init__(self):
        super().__init__("preview_upload", "预览上传")

    async def run(self, frames: List[Dict[str, Any]] = None, image_path: str = "", **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames, image_path)
        previews = []
        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            try:
                minio_url = MinioEngine.upload_file(path)
                previews.append(minio_url)
            except:
                img = cv2.imread(path)
                b64 = ImageUtils.encode_to_base64(img)
                previews.append(f"data:image/jpeg;base64,{b64}")
        return {"status": "success", "preview_images": previews}

class FaceIdentifyTool(BaseTool):
    def __init__(self):
        super().__init__("face_identify", "人脸识别")

    async def run(self, frames: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        persons, visual_risks, evidence_bboxes = [], [], []
        for item in frames:
            path = item.get("path")
            if not path or not os.path.exists(path): continue
            try:
                minio_url = item.get("minio_url") or MinioEngine.upload_file(path)
                results = await asyncio.to_thread(FaceEngine.identify_face, minio_url)
                if results:
                    for p in results:
                        p_name, p_tag, p_bbox = p.get("name", "未知"), p.get("tag", ""), p.get("bbox", [])
                        persons.append({"index": item["index"], "name": p_name, "tag": p_tag, "similarity": p.get("similarity", 0)})
                        if any(x in p_tag for x in ["黑名单", "敏感", "落马"]):
                             visual_risks.append(f"发现敏感人物: {p_name} ({p_tag})")
                             if p_bbox: evidence_bboxes.append({"frame_index": item["index"], "bbox": p_bbox, "color": (0, 0, 255)})
            except: pass
        return {"status": "success", "persons": persons, "visual_risks": visual_risks, "evidence_bboxes": evidence_bboxes}

class YoloDetectTool(BaseTool):
    def __init__(self):
        super().__init__("yolo_detect", "目标检测")

    async def run(self, frames: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        detections = []
        for item in frames:
            img = cv2.imread(item.get("path"))
            if img is None: continue
            raw = YoloEngine.detect(img)
            merged = ImageUtils.merge_overlapping_boxes(raw, img.shape)
            detections.append({"index": item["index"], "bboxes": merged})
        return {"status": "success", "detections": detections}

class OcrDetectTool(BaseTool):
    def __init__(self):
        super().__init__("ocr_detect", "OCR识别")

    async def run(self, frames: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        ocr_results = []
        for item in frames:
            img = cv2.imread(item.get("path"))
            if img is None: continue
            results = OcrEngine.detect_text(img)
            ocr_results.append({"index": item["index"], "path": item["path"], "items": results})
        return {"status": "success", "ocr_results": ocr_results}

class OcrRiskJudgeTool(BaseTool):
    def __init__(self):
        super().__init__("ocr_risk_judge", "OCR风险判定")

    async def run(self, ocr_results: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        if not ocr_results: return {"error": "无OCR数据"}
        risks, evidence_bboxes = [], []
        for item in ocr_results:
            texts_data = item.get("items", [])
            if not texts_data: continue
            full_text = " ".join([t.get("text", "") for t in texts_data])[:4000]
            text_map = {t.get("id", i): t.get("text", "") for i, t in enumerate(texts_data[:100])}
            prompt = PromptTemplates.ocr_judge_prompt(full_text, text_map)
            try:
                response_text = await LLMEngine.get_client().ainvoke(prompt, "")
                res = JSONUtils.safe_json_loads(response_text)
                if res and isinstance(res, dict) and res.get("id"):
                    hit_ids = [str(x) for x in (res["id"] if isinstance(res["id"], list) else [res["id"]])]
                    risks.append(f"发现敏感文字: {res.get('reason', '未知原因')}")
                    for t in texts_data:
                        if str(t.get("id")) in hit_ids:
                            pts = t.get("box")
                            if isinstance(pts, list) and len(pts) == 4:
                                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                                evidence_bboxes.append({"frame_index": item.get("index", 0), "bbox": [min(xs), min(ys), max(xs), max(ys)], "color": (0, 0, 255)})
            except: pass
        return {"status": "success", "ocr_risks": risks, "evidence_bboxes": evidence_bboxes}

class BehaviorJudgeTool(BaseTool):
    def __init__(self):
        super().__init__("behavior_judge", "行为判定")

    async def run(self, frames: List[Dict[str, Any]] = None, bboxes: List[List[int]] = None, **kwargs) -> Dict[str, Any]:
        frames = _normalize_frames_input(frames)
        visual_risks, evidence_bboxes = [], []
        for item in frames:
            img = cv2.imread(item.get("path"))
            if img is None: continue
            frame_bboxes = item.get("bboxes") or bboxes or []
            slices_b64, valid_bboxes = [], []
            for det in frame_bboxes:
                bb = det.get('bbox') if isinstance(det, dict) else det
                if bb:
                    slices_b64.append(ImageUtils.encode_to_base64(ImageUtils.get_single_object_crop(img, bb)))
                    valid_bboxes.append(bb)
            if not slices_b64: continue
            prompt = PromptTemplates.get_image_prompt("违规行为、敏感标识、阴暗内容、同性低俗、政治旗帜")
            try:
                res = await LLMEngine.get_json_response(LLMEngine.build_visual_message(prompt, slices_b64))
                if res and res.get("image"):
                    for idx in res["image"]:
                        if 0 < idx <= len(valid_bboxes):
                            bb = valid_bboxes[idx-1]
                            visual_risks.append(f"发现敏感内容 (帧 {item['index']}, ID {idx})")
                            evidence_bboxes.append({"frame_index": item["index"], "bbox": bb, "color": (0, 0, 255)})
            except: pass
        return {"status": "success", "visual_risks": visual_risks, "evidence_bboxes": evidence_bboxes}

class ImageAnnotateTool(BaseTool):
    def __init__(self): super().__init__("image_annotate", "（已停用，改用EvidenceUtils）")
    async def run(self, **kwargs): return {"status": "deprecated"}