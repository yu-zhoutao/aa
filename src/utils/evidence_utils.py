import cv2
import uuid
import os
import numpy as np
from typing import List, Dict, Any
from ..config import Config

class EvidenceUtils:
    """证据处理工具类：负责框合并、统一绘图"""

    @staticmethod
    def boxes_overlap(b1, b2):
        """判断两个矩形框是否重叠"""
        # b1, b2格式: [x1, y1, x2, y2]
        return max(b1[0], b2[0]) < min(b1[2], b2[2]) and \
               max(b1[1], b2[1]) < min(b1[3], b2[3])

    @staticmethod
    def merge_overlapping_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并重叠的框
        输入: [{'bbox': [x1,y1,x2,y2], 'label': 'xxx', 'color': (r,g,b)}, ...]"""
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

        # 简单的重叠合并逻辑
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
            # 合并坐标
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            
            # 合并标签 (去重)
            labels = set()
            for c in cluster:
                if c.get("label"): labels.add(c["label"])
            
            merged.append({
                "bbox": [x1, y1, x2, y2],
                "label": "|".join(labels),
                "color": cluster[0].get("color", (0, 0, 255)) # 沿用第一个框的颜色
            })
        return merged

    @staticmethod
    def generate_evidence_image(original_image_path: str, bboxes: List[Dict[str, Any]]) -> str:
        """
        在原图上绘制所有bbox，并保存为一张新图
        :param bboxes: [{'bbox': [x1,y1,x2,y2], 'label': '...', 'color': (0,0,255)}]"""
        if not os.path.exists(original_image_path):
            return ""
            
        img = cv2.imread(original_image_path)
        if img is None: return ""

        # 合并框
        final_bboxes = EvidenceUtils.merge_overlapping_boxes(bboxes)
        
        for item in final_bboxes:
            x1, y1, x2, y2 = map(int, item['bbox'])
            color = item.get('color', (0, 0, 255)) # 默认红色
            label = item.get('label', '')
            
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # 画标签背景
            if label:
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        config = Config()
        filename = f"evidence_final_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(config.temp_dir, filename)
        cv2.imwrite(save_path, img)
        
        return save_path
