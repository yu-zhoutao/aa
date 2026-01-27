
import os
import hashlib
from minio import Minio
from info_judge_next.config import Config

class MinioEngine:
    """MinIO Engine with Deduplication"""
    
    _client = None
    _url_cache = {} # Local path -> URL cache

    @classmethod
    def get_client(cls) -> Minio:
        if cls._client is None:
            cls._client = Minio(
                Config.MINIO_ENDPOINT,
                access_key=Config.MINIO_ACCESS_KEY,
                secret_key=Config.MINIO_SECRET_KEY,
                secure=Config.MINIO_SECURE
            )
        return cls._client

    @classmethod
    def _calculate_md5(cls, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @classmethod
    def upload_file(cls, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path in cls._url_cache:
            return cls._url_cache[file_path]

        client = cls.get_client()
        bucket_name = Config.MINIO_BUCKET
        
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        file_hash = cls._calculate_md5(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        object_name = f"image/{file_hash}{ext}"
        
        protocol = "https" if Config.MINIO_SECURE else "http"
        url = f"{protocol}://{Config.MINIO_ENDPOINT}/{bucket_name}/{object_name}"
        
        try:
            client.stat_object(bucket_name, object_name)
            # Already exists
            url = url.replace("minio.di.qihoo.net:9000", "nrsh.di.360.cn")
            cls._url_cache[file_path] = url
            return url
        except:
            pass

        content_type = "application/octet-stream"
        if ext in ['.jpg', '.jpeg']:
            content_type = "image/jpeg"
        elif ext == '.png':
            content_type = "image/png"
        
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type
        )
        url = url.replace("minio.di.qihoo.net:9000", "nrsh.di.360.cn")
        cls._url_cache[file_path] = url
        return url
