import os
import shutil
import time
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from ..config import Config
from ..engines.minio_engine import MinioEngine
import aiohttp

class FileUtils:
    """文件系统操作工具类"""

    @staticmethod
    def detect_file_type(filename: str) -> str:
        """
        根据扩展名探测媒体类型
        """
        ext = Path(filename).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
            return "image"
        if ext in ['.mp3', '.wav', '.aac', '.flac', '.m4a']:
            return "audio"
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
            return "video"
        return "unknown"

    @staticmethod
    def _calculate_md5_from_upload(upload_file) -> str:
        """
        计算 FastAPI UploadFile 对象的 MD5 值
        """
        hash_md5 = hashlib.md5()
        original_position = upload_file.file.tell()
        
        try:
            upload_file.file.seek(0)
            for chunk in iter(lambda: upload_file.file.read(4096), b""):
                hash_md5.update(chunk)
            upload_file.file.seek(0)
            return hash_md5.hexdigest()
        except Exception as e:
            upload_file.file.seek(original_position)
            raise e

    @staticmethod
    def save_upload_file(upload_file, custom_name: Optional[str] = None, upload_to_minio: bool = True) -> Tuple[str, Optional[str]]:
        """
        将 FastAPI 的 UploadFile 对象保存到临时目录，并可选地上传到 MinIO
        """
        config = Config()
        if not os.path.exists(config.temp_dir):
            os.makedirs(config.temp_dir)
            
        ext = Path(upload_file.filename).suffix
        
        if custom_name is None:
            try:
                file_hash = FileUtils._calculate_md5_from_upload(upload_file)
                filename = f"{file_hash}{ext}"
            except Exception as e:
                print(f"⚠️ 计算 MD5 失败: {str(e)}")
                filename = f"{uuid.uuid4().hex}{ext}"
        else:
            filename = custom_name
        
        file_path = os.path.join(config.temp_dir, filename)
        
        # 保存到本地
        try:
            # seek(0) 确保从头开始读取，防止之前计算MD5导致指针在文件末尾
            upload_file.file.seek(0)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
        except Exception as e:
            print(f"❌ 本地保存失败: {e}")
            raise e
        
        # 上传到 MinIO
        minio_url = None
        if upload_to_minio:
            try:
                minio_url = MinioEngine.upload_file(file_path)
                print(f"✅ 文件已上传到 MinIO: {minio_url}")
            except Exception as e:
                print(f"⚠️ 文件上传到 MinIO 失败: {str(e)}")
            
        return file_path, minio_url

    @staticmethod
    def clear_temp_dir(age_seconds: int = 3600):
        """
        清理临时目录中超过一定时间的文件
        """
        config = Config()
        now = time.time()
        if not os.path.exists(config.temp_dir):
            return

        for item in os.listdir(config.temp_dir):
            item_path = os.path.join(config.temp_dir, item)
            if os.path.getmtime(item_path) < now - age_seconds:
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    print(f"🧹 已自动清理过期文件: {item}")
                except Exception as e:
                    print(f"❌ 清理文件失败 {item}: {e}")

    @staticmethod
    def get_static_url(file_path: str) -> str:
        """
        将本地路径转换为前端可访问的静态 URL 路径
        """
        filename = os.path.basename(file_path)
        return f"/static_temp/{filename}"

    @staticmethod
    async def async_serper_search(image_url: str, extra_query: str = "") -> str:
        if not image_url or not Config.SERPAPI_KEY: return "未启用搜索。"
        
        # 1. 修改参数适配 Google Lens
        params = {
            "engine": "google_lens", 
            "url": image_url, 
            "api_key": Config.SERPAPI_KEY, 
            "hl": "zh-CN", 
            "gl": "cn"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search.json", params=params) as response:
                    if response.status != 200:
                        return f"搜索请求失败，状态码: {response.status}"
                    data = await response.json()
            
            results_text = []
            
            # 2. 优先提取知识图谱（如果识别出了具体物体/人物）
            if "knowledge_graph" in data:
                kg = data['knowledge_graph']
                kg_title = kg.get('title', '')
                if kg_title:
                    results_text.append(f"【识别结果】: {kg_title}")
            
            # 3. 提取 visual_matches 中的 title 和 link
            matches = data.get("visual_matches", [])
            
            for item in matches[:6]: # 限制返回前 6 个
                title = item.get("title", "").strip()
                link = item.get("link", "")
                source = item.get("source", "")
                
                if title and link:
                    # 格式：[来源] 标题 + 换行链接
                    entry = f"- [{source}] {title}\n  {link}" if source else f"- {title}\n  {link}"
                    results_text.append(entry)
                
            return "\n".join(results_text) if results_text else "未搜索到相关结果。"
            
        except Exception as e:
            return f"搜索服务错误: {str(e)}"