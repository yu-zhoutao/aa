
import torch
from ultralytics import YOLO
from typing import List, Dict, Any
from info_judge_next.config import Config

class YoloEngine:
    _model = None

    @classmethod
    def get_model(cls) -> YOLO:
        if cls._model is None:
            print(f"🚀 Loading YOLO: {Config.YOLO_MODEL_PATH} ...")
            cls._model = YOLO(Config.YOLO_MODEL_PATH).to(Config.DEVICE)
        return cls._model

    @classmethod
    def detect(cls, image_path_or_array: Any, conf: float = 0.25) -> List[Dict[str, Any]]:
        model = cls.get_model()
        results = model(image_path_or_array, conf=conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None: continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls_idx in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls_idx]
                detections.append({
                    "label": label,
                    "conf": float(score),
                    "bbox": [x1, y1, x2, y2]
                })
        return detections
