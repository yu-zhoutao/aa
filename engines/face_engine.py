
import requests
from typing import List, Dict, Any
from info_judge_next.config import Config

class FaceEngine:
    @staticmethod
    def identify_face(image_url: str) -> List[Dict[str, Any]]:
        url = Config.FACE_API_URL
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
            print(f"🚀 Calling Face API: {image_url}")
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
                                first_class = info.get("first_class", "")
                                second_class = info.get("second_class", "")
                                similarity = info.get("similarity", 0)
                                bbox = info.get("bbox", [])
                                found_results.append({
                                    "name": name,
                                    "tag": f"{first_class} | {second_class}",
                                    "similarity": similarity,
                                    "bbox": bbox
                                })
            else:
                print(f"❌ Face API Failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Face API Error: {e}")
            
        return found_results
