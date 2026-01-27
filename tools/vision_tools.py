
import asyncio
from typing import Dict, List, Any
from info_judge_next.tools.base import BaseTool
from info_judge_next.engines.ocr_engine import OcrEngine
from info_judge_next.engines.face_engine import FaceEngine
from info_judge_next.engines.yolo_engine import YoloEngine

class OcrTool(BaseTool):
    name = "ocr_scan"
    description = "使用 OCR 检测图片中的文字。"

    async def run(self, file_path: str) -> Dict[str, Any]:
        try:
            results = OcrEngine.detect_text(file_path)
            return {"status": "success", "text_detections": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_args_schema(self) -> Dict:
        return {"file_path": {"type": "string", "description": "图片文件的本地路径"}}
    def _get_required_args(self) -> List[str]:
        return ["file_path"]

class FaceDetectionTool(BaseTool):
    name = "face_scan"
    description = "使用 MinIO URL (必须) 检测图片中的人脸。"

    async def run(self, image_url: str) -> Dict[str, Any]:
        try:
            # FaceEngine 需要 URL，而非本地路径
            results = await asyncio.to_thread(FaceEngine.identify_face, image_url)
            return {"status": "success", "face_detections": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_args_schema(self) -> Dict:
        return {"image_url": {"type": "string", "description": "图片的 MinIO URL"}}
    def _get_required_args(self) -> List[str]:
        return ["image_url"]

class YoloDetectionTool(BaseTool):
    name = "object_detection"
    description = "使用 YOLO 检测图片中的物体（如旗帜、人物）。"

    async def run(self, file_path: str) -> Dict[str, Any]:
        try:
            results = YoloEngine.detect(file_path, conf=0.3)
            return {"status": "success", "object_detections": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_args_schema(self) -> Dict:
        return {"file_path": {"type": "string", "description": "图片文件的本地路径"}}
    def _get_required_args(self) -> List[str]:
        return ["file_path"]
