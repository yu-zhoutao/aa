
import base64
import ast
import requests
import numpy as np
import cv2
from typing import List, Dict, Any, Union
import urllib3
from info_judge_next.config import Config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class OcrEngine:
    @classmethod
    def _encode_image(cls, image_source: Union[str, np.ndarray]) -> str:
        img_data = None
        if isinstance(image_source, str):
            with open(image_source, "rb") as f:
                img_data = f.read()
        elif isinstance(image_source, np.ndarray):
            success, encoded_img = cv2.imencode('.jpg', image_source)
            if not success:
                raise ValueError("Cannot encode numpy array")
            img_data = encoded_img.tobytes()
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")
        return base64.b64encode(img_data).decode("utf-8")

    @classmethod
    def detect_text(cls, image_source: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        ocr_results = []
        try:
            encoded_image = cls._encode_image(image_source)
            url = Config.OCR_API_URL
            payload = {"IMAGE": encoded_image, "base64_list": ["IMAGE"]}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {Config.OCR_API_KEY}"}

            response = requests.post(url, headers=headers, json=payload, timeout=30, verify=False, proxies={"http": None, "https": None})

            if response.ok:
                outer_response = response.json()
                if "bridge_output0" in outer_response:
                    bridge_output = outer_response["bridge_output0"]
                    if bridge_output:
                        output = ast.literal_eval(bridge_output)
                        extra_bbox = output.get("extra_bbox", [])
                        extra_info = output.get("extra_info", [])

                        for idx, (box, text) in enumerate(zip(extra_bbox, extra_info)):
                            ocr_results.append({
                                "id": idx + 1,
                                "text": text,
                                "box": box
                            })
        except Exception as e:
            print(f"❌ OCR Error: {e}")

        return ocr_results
