
import os
import asyncio
from typing import Dict, List, Any, Union
from info_judge_next.tools.base import BaseTool
from info_judge_next.engines.minio_engine import MinioEngine
from info_judge_next.utils.image_utils import ImageUtils

class MinioUploadTool(BaseTool):
    name = "upload_to_minio"
    description = "将本地文件上传到 MinIO 存储并获取 URL。支持批量上传。自动检查重复文件。"

    async def run(self, file_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        接收单个路径字符串或路径字符串列表。
        返回 file_path -> url 的映射。
        """
        paths = []
        if isinstance(file_paths, str):
            paths = [file_paths]
        elif isinstance(file_paths, list):
            paths = file_paths
        else:
            return {"status": "error", "message": "参数类型无效。期望字符串或字符串列表。"}

        results = {}
        errors = []

        # 并发上传
        async def _upload_one(p):
            try:
                # 使用 to_thread 处理阻塞 IO/网络调用
                url = await asyncio.to_thread(MinioEngine.upload_file, p)
                return p, url, None
            except Exception as e:
                return p, None, str(e)

        tasks = [_upload_one(p) for p in paths]
        if not tasks:
             return {"status": "success", "uploaded_files": {}}

        upload_results = await asyncio.gather(*tasks)

        for p, url, err in upload_results:
            if url:
                results[p] = url
            else:
                errors.append(f"{p}: {err}")

        return {
            "status": "success" if not errors else "partial_success",
            "uploaded_files": results, # Map: local_path -> minio_url
            "errors": errors
        }

    def _get_args_schema(self) -> Dict:
        return {
            "file_paths": {
                "anyOf": [
                    {"type": "string", "description": "单个本地绝对路径"},
                    {"type": "array", "items": {"type": "string"}, "description": "本地绝对路径列表"}
                ],
                "description": "文件的本地绝对路径"
            }
        }
    def _get_required_args(self) -> List[str]:
        return ["file_paths"]

class FrameExtractionTool(BaseTool):
    name = "extract_frames"
    description = "从视频文件提取帧，或加载图像文件。"

    async def run(self, file_path: str) -> Dict[str, Any]:
        frames = ImageUtils.extract_frames(file_path)
        # 我们需要将帧保存到临时文件以返回路径
        results = []
        import uuid
        from info_judge_next.config import Config
        import cv2

        for item in frames:
            temp_name = f"{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(Config.FIXED_TEMP_DIR, temp_name)
            cv2.imwrite(temp_path, item['img'])
            results.append({
                "index": item['index'],
                "file_path": temp_path
            })
        
        return {"status": "success", "frames": results}

    def _get_args_schema(self) -> Dict:
        return {
            "file_path": {"type": "string", "description": "视频或图片文件的路径"}
        }
    def _get_required_args(self) -> List[str]:
        return ["file_path"]
