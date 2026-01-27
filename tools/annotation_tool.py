
import cv2
import os
import uuid
from typing import Dict, List, Any
from info_judge_next.tools.base import BaseTool
from info_judge_next.config import Config
from info_judge_next.utils.image_utils import ImageUtils

class ImageAnnotationTool(BaseTool):
    name = "annotate_image"
    description = "在图片上绘制边界框和标签。支持多种类型的检测（人脸、OCR、物体）。"

    async def run(self, file_path: str, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        annotations: 字典列表，例如：
        [
          {"bbox": [x1, y1, x2, y2], "label": "Text", "type": "ocr"},
          {"bbox": [x1, y1, x2, y2], "label": "Person", "type": "face"},
          {"bbox": [x1, y1, x2, y2], "label": "Flag", "type": "object"}
        ]
        """
        if not os.path.exists(file_path):
            return {"status": "error", "message": "文件未找到"}

        img = cv2.imread(file_path)
        if img is None:
            return {"status": "error", "message": "无法加载图片"}

        for item in annotations:
            bbox = item.get("bbox")
            label = item.get("label", "")
            type_ = item.get("type", "unknown")
            
            # 颜色编码
            if type_ == "ocr":
                color = (0, 255, 0) # 绿色
            elif type_ == "face":
                color = (0, 0, 255) # 红色
            elif type_ == "violation":
                 color = (0, 0, 255) # 红色
            else:
                color = (255, 0, 0) # 蓝色

            # 绘制
            if bbox:
                # 确保 bbox 格式
                 if len(bbox) == 4:
                     # 检查是 OCR 四点还是 xyxy
                     # OCR 可能在某些引擎中作为 box 传递，但这里为简单起见我们期望 [x1,y1,x2,y2]
                     # 如果 LLM 传递原始 OCR box（4点），我们需要处理它。
                     # 但假设 Agent 将其标准化或我们处理简单的 xyxy。
                     x1, y1, x2, y2 = map(int, bbox)
                     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                     if label:
                         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存到新文件
        new_filename = f"annotated_{uuid.uuid4().hex}.jpg"
        new_path = os.path.join(Config.FIXED_TEMP_DIR, new_filename)
        cv2.imwrite(new_path, img)
        
        # 返回路径和 base64 预览
        b64 = ImageUtils.encode_to_base64(img)
        
        return {
            "status": "success", 
            "annotated_file_path": new_path,
            "preview_base64": b64
        }

    def _get_args_schema(self) -> Dict:
        return {
            "file_path": {"type": "string", "description": "原图片路径"},
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {"type": "array", "items": {"type": "integer"}, "description": "[x1, y1, x2, y2]"},
                        "label": {"type": "string", "description": "标签文本"},
                        "type": {"type": "string", "enum": ["ocr", "face", "object", "violation"], "description": "检测类型"}
                    },
                    "required": ["bbox"]
                },
                "description": "标注列表"
            }
        }
    def _get_required_args(self) -> List[str]:
        return ["file_path", "annotations"]
