import cv2
import uuid
import os
import numpy as np
from typing import List, Dict, Any
from ..config import Config

class EvidenceUtils:
    """证据处理工具类：负责框合并、纯矩形框绘图"""

    @staticmethod
    def boxes_overlap(b1, b2):
        """判断两个矩形框是否重叠"""
        return max(b1[0], b2[0]) < min(b1[2], b2[2]) and \
               max(b1[1], b2[1]) < min(b1[3], b2[3])

    @staticmethod
    def merge_overlapping_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并重叠的框
        """
        if not detections: return []
        
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
                if EvidenceUtils.boxes_overlap(detections[i]["bbox"], detections[j]["bbox"]):
                    union(i, j)

        clusters = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(detections[i])

        merged = []
        for cluster in clusters.values():
            boxes = [c["bbox"] for c in cluster]
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            
            merged.append({
                "bbox": [x1, y1, x2, y2],
                "color": cluster[0].get("color", (0, 0, 255))
            })
        return merged

    @staticmethod
    def generate_evidence_image(original_image_path: str, bboxes: List[Dict[str, Any]]) -> str:
        """
        在原图上绘制纯矩形框，并保存为一张新图
        """
        if not os.path.exists(original_image_path):
            return ""
            
        img = cv2.imread(original_image_path)
        if img is None: return ""

        # 合并重叠框
        final_bboxes = EvidenceUtils.merge_overlapping_boxes(bboxes)
        
        for item in final_bboxes:
            x1, y1, x2, y2 = map(int, item['bbox'])
            color = item.get('color', (0, 0, 255)) # 默认红色
            
            # 仅绘制矩形框，不写字
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        config = Config()
        # 使用更固定的命名前缀，方便识别
        filename = f"evidence_final_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(config.temp_dir, filename)
        cv2.imwrite(save_path, img)
        
        return save_path