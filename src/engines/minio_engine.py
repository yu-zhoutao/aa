import os
import hashlib
from minio import Minio
from ..utils.config import Config

class MinioEngine:
    """MinIO 文件存储引擎 (支持内容去重)"""
    
    _client = None
    _url_cache = {} # 本地路径 -> URL 缓存

    @classmethod
    def get_client(cls) -> Minio:
        config = Config()
        if cls._client is None:
            cls._client = Minio(
                config.minio_endpoint,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                secure=config.minio_secure
            )
        return cls._client

    @classmethod
    def _calculate_md5(cls, file_path: str) -> str:
        """计算文件的 MD5 值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @classmethod
    def _get_content_type(cls, ext: str) -> str:
        """根据文件扩展名获取 Content-Type"""
        content_type = "application/octet-stream"
        
        if ext in ['.jpg', '.jpeg']: content_type = "image/jpeg"
        elif ext == '.png': content_type = "image/png"
        elif ext in ['.gif', '.webp', '.bmp']: content_type = f"image/{ext[1:]}"
        elif ext in ['.mp3']: content_type = "audio/mpeg"
        elif ext == '.wav': content_type = "audio/wav"
        elif ext == '.mp4': content_type = "video/mp4"
        
        return content_type

    @classmethod
    def _get_storage_path(cls, ext: str) -> str:
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']: return "image"
        elif ext in ['.mp3', '.wav', '.aac', '.flac', '.m4a']: return "audio"
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']: return "video"
        else: return "other"

    @classmethod
    def upload_file(cls, file_path: str) -> str:
        """
        上传文件到 MinIO (带去重逻辑)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if file_path in cls._url_cache:
            return cls._url_cache[file_path]

        client = cls.get_client()
        config = Config()
        bucket_name = config.minio_bucket
        
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        file_hash = cls._calculate_md5(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        storage_path = cls._get_storage_path(ext)
        object_name = f"{storage_path}/{file_hash}{ext}"
        
        protocol = "https" if config.minio_secure else "http"
        url = f"{protocol}://{config.minio_endpoint}/{bucket_name}/{object_name}"
        
        try:
            client.stat_object(bucket_name, object_name)
            cls._url_cache[file_path] = url
            return url
        except:
            pass

        content_type = cls._get_content_type(ext)
        
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type
        )

        cls._url_cache[file_path] = url
        return url
