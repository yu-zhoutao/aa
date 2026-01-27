
import os
import torch
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # --- API ---
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    VLLM_API_URL = os.getenv("API_URL", "http://127.0.0.1:8008/v1") 
    VLLM_API_KEY = os.getenv("API_KEY", "EMPTY") 
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct") 
    
    # --- MinIO ---
    MINIO_ENDPOINT = "minio.di.qihoo.net:9000"
    MINIO_ACCESS_KEY = "zhangshuhao"
    MINIO_SECRET_KEY = "MinIO@2025.qihoo"
    MINIO_BUCKET = "facerun-content-detect"
    MINIO_SECURE = False

    # --- Face API ---
    FACE_API_URL = "http://hpcinf01.aitc.bjwdt.qihoo.net:6980/api/v1/image/sync"

    # --- Models ---
    YOLO_MODEL_PATH = "./yolov8n.pt"

    # --- OCR ---
    OCR_API_URL = os.getenv("OCR_API_URL")
    OCR_API_KEY = os.getenv("OCR_API_KEY")

    # --- ASR (Speech-to-Text) ---
    ASR_API_URL = os.getenv("ASR_API_URL")
    ASR_API_KEY = os.getenv("ASR_API_KEY")
    ASR_THREAD_POOL_SIZE = 6

    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    FIXED_TEMP_DIR = os.path.join(BASE_DIR, "upload_cache")

    # --- Device ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def init_directories(cls):
        if not os.path.exists(cls.FIXED_TEMP_DIR):
            os.makedirs(cls.FIXED_TEMP_DIR)
            print(f"📁 Created directory: {cls.FIXED_TEMP_DIR}")

Config.init_directories()
