
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional
from judge_agent.config import Config
import aiohttp

class FileUtils:
    """File system utilities"""

    @staticmethod
    def detect_file_type(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
            return "image"
        if ext in ['.mp3', '.wav', '.aac', '.flac', '.m4a']:
            return "audio"
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
            return "video"
        return "unknown"

    @staticmethod
    def save_upload_file(upload_file, custom_name: Optional[str] = None) -> str:
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            os.makedirs(Config.FIXED_TEMP_DIR)
            
        ext = Path(upload_file.filename).suffix
        filename = custom_name or f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(Config.FIXED_TEMP_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return file_path

    @staticmethod
    def clear_temp_dir(age_seconds: int = 3600):
        now = time.time()
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            return

        for item in os.listdir(Config.FIXED_TEMP_DIR):
            item_path = os.path.join(Config.FIXED_TEMP_DIR, item)
            if os.path.getmtime(item_path) < now - age_seconds:
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"❌ Failed to clear {item}: {e}")

    @staticmethod
    def get_static_url(file_path: str) -> str:
        filename = os.path.basename(file_path)
        return f"/static_temp/{filename}"

    @staticmethod
    async def async_serper_search(image_url: str, extra_query: str = "") -> str:
        if not image_url or not Config.SERPAPI_KEY: return "Search not enabled."
        
        params = {
            "engine": "google_reverse_image", "image_url": image_url,
            "api_key": Config.SERPAPI_KEY, "hl": "zh-CN", "gl": "cn"
        }
        if extra_query: params["q"] = extra_query
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search.json", params=params) as response:
                    data = await response.json()
            
            results_text = []
            if "knowledge_graph" in data:
                results_text.append(f"【Knowledge Graph】: {data['knowledge_graph'].get('title', '')}")
            
            results = data.get("image_results", []) + data.get("inline_images", [])
            for item in results[:6]:
                title = item.get("title", "")
                source = item.get("source", "")
                if title: results_text.append(f"- [{source}] {title}")
                
            return "\n".join(results_text) if results_text else "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"
