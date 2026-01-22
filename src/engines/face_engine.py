import requests
from typing import List, Dict, Any
from ..utils.config import Config

class FaceEngine:
    """API based Face Recognition Engine"""

    @staticmethod
    def identify_face(image_url: str) -> List[Dict[str, Any]]:
        config = Config()
        url = config.face_api_url
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        payload = {
            "ability": ["face"],
            "tasks": [{"dataId": "audit_task", "url": image_url}],
            "rule": []
        }
        
        found_results = []
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("code") == 200:
                    results_list = res_json.get("result", [])
                    for task_res in results_list:
                        face_data = task_res.get("face", {})
                        detail = face_data.get("detail", {})
                        extra_info = detail.get("extra_info", [])
                        
                        for info in extra_info:
                            name = info.get("name")
                            if name and name != "unknown":
                                found_results.append({
                                    "name": name,
                                    "tag": f"{info.get('first_class','')} | {info.get('second_class','')}",
                                    "similarity": info.get("similarity", 0),
                                    "bbox": info.get("bbox", [])
                                })
            else:
                print(f"❌ Face API 请求失败: {response.status_code}")
        except Exception as e:
            print(f"❌ Face API 请求异常: {e}")
            
        return found_results
