import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # LLM 配置
    api_url: str = os.getenv("API_URL", "http://127.0.0.1:8008/v1")
    api_key: str = os.getenv("API_KEY", "EMPTY")
    model_name: str = os.getenv("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct")
    
    # MinIO 配置
    minio_endpoint: str = "minio.di.qihoo.net:9000"
    minio_access_key: str = "zhangshuhao"
    minio_secret_key: str = "MinIO@2025.qihoo"
    minio_bucket: str = "facerun-content-detect"
    minio_secure: bool = False

    # 临时文件目录
    temp_dir: str = os.getenv("TEMP_DIR", "upload_cache")
    
    # 工具 API 配置
    face_api_url: str = "http://hpcinf01.aitc.bjwdt.qihoo.net:6980/api/v1/image/sync"
    
    # ASR
    asr_api_url: str = os.getenv("ASR_API_URL")
    asr_api_key: str = os.getenv("ASR_API_KEY", "")
    asr_thread_pool_size: int = int(os.getenv("ASR_THREAD_POOL_SIZE", "6"))
    
    # OCR
    ocr_api_url: str = os.getenv("OCR_API_URL")
    ocr_api_key: str = os.getenv("OCR_API_KEY", "")

    # Search API
    serpapi_key: str = os.getenv("SERPAPI_KEY", "")

def load_config() -> Config:
    return Config()
