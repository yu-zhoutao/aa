import cv2
import base64
import numpy as np
from typing import List, Dict, Any

class ImageUtils:
    """图像处理与视觉标注工具类"""

    @staticmethod
    def extract_frames(file_path: str, sample_count: int = 8) -> List[Dict]:
        """
        统一抽帧逻辑
        :param file_path: 文件路径
        :param sample_count: 视频采样帧数
        :return: [{'img': np.ndarray, 'index': int}, ...] 
        """
        import os
        frames = []
        if not os.path.exists(file_path):
            return []
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.png', '.jpeg', '.webp', '.bmp']:
            img = cv2.imread(file_path)
            if img is not None:
                frames.append({"img": img, "index": 0})
        else:
            # 视频
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return []
                
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(total // sample_count, 1)
            
            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append({"img": frame, "index": i})
                if len(frames) >= sample_count:
                    break
            cap.release()
        
        return frames

    @staticmethod
    def encode_to_base64(image: np.ndarray, quality: int = 90) -> str:
        """
        将 OpenCV 的 BGR 图像转换为 Base64 编码的 JPEG 字符串
        """
        if image is None:
            return ""
        success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def decode_from_base64(base64_str: str) -> np.ndarray:
        """
        将 Base64 字符串解码回 OpenCV 格式的 BGR 图像
        """
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], color=(0, 0, 255), thickness=2) -> np.ndarray:
        """
        在图像上绘制 YOLO 检测到的目标框
        """
        if image is None:
            return None
            
        temp_img = image.copy()
        for det in detections:
            bbox = det.get('bbox')
            if not bbox: continue
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, thickness)
            
        return temp_img

    @staticmethod
    def draw_ocr_boxes(image: np.ndarray, ocr_results: List[Dict[str, Any]], color=(0, 255, 0)) -> np.ndarray:
        """
        在图像上绘制 OCR 文字区域多边形
        """
        temp_img = image.copy()
        for ocr in ocr_results:
            box = ocr.get('box')
            if not box: continue
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_img, [pts], isClosed=True, color=color, thickness=2)
        return temp_img

    @staticmethod
    def get_single_object_crop(image: np.ndarray, bbox: List[int], padding: int = 10) -> np.ndarray:
        """
        根据 bbox 裁剪出单个目标区域
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(w, x2 + padding)
        y2_p = min(h, y2 + padding)
        return image[y1_p:y2_p, x1_p:x2_p]
    
    @staticmethod
    def boxes_overlap(b1, b2):
        return max(b1[0], b2[0]) < min(b1[2], b2[2]) and \
               max(b1[1], b2[1]) < min(b1[3], b2[3])

    @staticmethod
    def merge_overlapping_boxes(detections, img_shape):
        if not detections: return []
        h, w = img_shape[:2]
        n = len(detections)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[ry] = rx

        for i in range(n):
            for j in range(i + 1, n):
                if ImageUtils.boxes_overlap(detections[i]["bbox"], detections[j]["bbox"]):
                    union(i, j)

        clusters = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(detections[i])

        merged = []
        for idx, cluster in enumerate(clusters.values()):
            boxes = [c["bbox"] for c in cluster]
            x1 = max(0, min(b[0] for b in boxes))
            y1 = max(0, min(b[1] for b in boxes))
            x2 = min(w, max(b[2] for b in boxes))
            y2 = min(h, max(b[3] for b in boxes))
            
            main_label = cluster[0].get("label", "unknown")
            merged.append({
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2],
                "label": main_label
            })
        return merged
