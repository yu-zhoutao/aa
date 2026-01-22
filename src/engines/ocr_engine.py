import base64
import json
import ast
import requests
import numpy as np
import cv2
from typing import List, Dict, Any, Union
import urllib3
from ..config import Config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class OcrEngine:
    """在线 OCR 文字识别引擎"""

    @classmethod
    def _encode_image(cls, image_source: Union[str, np.ndarray]) -> str:
        img_data = None
        if isinstance(image_source, str):
            with open(image_source, "rb") as f:
                img_data = f.read()
        elif isinstance(image_source, np.ndarray):
            success, encoded_img = cv2.imencode('.jpg', image_source)
            if not success:
                raise ValueError("无法将 Numpy 数组编码为图像")
            img_data = encoded_img.tobytes()
        else:
            raise TypeError(f"不支持的图像类型: {type(image_source)}")
        return base64.b64encode(img_data).decode("utf-8")

    @classmethod
    def detect_text(cls, image_source: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        ocr_results = []
        config = Config()
        
        if not config.ocr_api_url:
            print("⚠️ OCR_API_URL 未配置，跳过 OCR 检测")
            return []

        try:
            encoded_image = cls._encode_image(image_source)
            payload = {
                "IMAGE": encoded_image,
                "base64_list": ["IMAGE"]
            }
            
            headers = {"Content-Type": "application/json"}
            if config.ocr_api_key:
                headers["Authorization"] = f"Bearer {config.ocr_api_key}"
            
            response = requests.post(
                config.ocr_api_url, 
                headers=headers, 
                json=payload, 
                timeout=30, 
                verify=False,
                proxies={"http": None, "https": None}
            )

            if response.ok:
                outer_response = response.json()
                if "bridge_output0" in outer_response:
                    bridge_output = outer_response["bridge_output0"]
                    if bridge_output:
                        try:
                            output = ast.literal_eval(bridge_output)
                        except:
                            # 尝试修复 JSON 格式问题或改用 json.loads
                            output = {}
                            
                        extra_bbox = output.get("extra_bbox", [])
                        extra_info = output.get("extra_info", [])

                        for idx, (box, text) in enumerate(zip(extra_bbox, extra_info)):
                            ocr_results.append({
                                "id": idx + 1,
                                "text": text,
                                "box": box
                            })
            else:
                print(f"❌ OCR API 请求失败: {response.status_code}")

        except Exception as e:
            print(f"❌ OCR 识别错误: {e}")

        return ocr_results