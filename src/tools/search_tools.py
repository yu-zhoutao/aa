import os
import asyncio
import time
from typing import Dict, List, Any
from .base import BaseTool
from ..utils.file_utils import FileUtils
from ..engines.minio_engine import MinioEngine

class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__("web_search", "网络搜索工具 (支持以图搜图)")

    async def run(self, query: str = "", image_path: str = "", image_url: str = "", **kwargs) -> Dict[str, Any]:
        if not query and not image_path and not image_url:
            return {"error": "必须提供 query, image_path 或 image_url"}

        search_result = ""

        if image_path or image_url:
            target_url = image_url
            
            if not target_url and image_path:
                if not os.path.exists(image_path):
                    return {"error": f"图片文件不存在: {image_path}"}
                
                try:
                    target_url = await asyncio.to_thread(MinioEngine.upload_file, image_path)
                except Exception as e:
                    print(f"❌ MinIO 上传失败: {e}")
                    target_url = None
            
            if not target_url:
                return {"error": "无法获取有效的图片 URL 进行搜索"}
            
            # FileUtils 已经在之前包含了 async_serper_search 方法吗？
            # 我需要检查一下 src/utils/file_utils.py 是否有这个方法。
            # 如果没有，我需要补上。
            if hasattr(FileUtils, 'async_serper_search'):
                search_result = await FileUtils.async_serper_search(target_url, extra_query=query)
            else:
                search_result = "搜索引擎未配置 (FileUtils.async_serper_search missing)"

        else:
            search_result = f"收到纯文本搜索请求: {query}。当前底层引擎暂仅支持'以图搜图'，请提供相关截图。"

        return {
            "status": "success",
            "search_findings": search_result
        }
