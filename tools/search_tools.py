
import asyncio
from typing import Dict, List, Any, Union
from info_judge_next.tools.base import BaseTool
from info_judge_next.utils.file_utils import FileUtils

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "使用 MinIO URL 执行反向图像搜索（以图搜图）。支持批量搜索。"

    async def run(self, image_urls: Union[str, List[str]], query: str = "") -> Dict[str, Any]:
        """
        接收单个 URL 字符串或 URL 字符串列表。
        返回搜索结果。
        """
        urls = []
        if isinstance(image_urls, str):
            urls = [image_urls]
        elif isinstance(image_urls, list):
            urls = image_urls
        else:
            return {"status": "error", "message": "参数类型无效。期望字符串或字符串列表。"}

        if not urls:
             return {"status": "error", "message": "未提供图片 URL。"}

        # 并发搜索
        async def _search_one(u):
            try:
                res = await FileUtils.async_serper_search(u, extra_query=query)
                return u, res, None
            except Exception as e:
                return u, None, str(e)

        tasks = [_search_one(u) for u in urls]
        search_results = await asyncio.gather(*tasks)

        findings = {}
        errors = []
        
        agg_text = []

        for u, res, err in search_results:
            if res:
                findings[u] = res
                agg_text.append(f"[图片: {u[-20:]}...] 结果: {res}")
            else:
                errors.append(f"{u}: {err}")

        return {
            "status": "success" if not errors else "partial_success",
            "search_findings": "\n\n".join(agg_text),
            "details": findings,
            "errors": errors
        }

    def _get_args_schema(self) -> Dict:
        return {
            "image_urls": {
                "anyOf": [
                    {"type": "string", "description": "单个 MinIO URL"},
                    {"type": "array", "items": {"type": "string"}, "description": "MinIO URL 列表"}
                ],
                "description": "要搜索的图片 URL"
            },
            "query": {"type": "string", "description": "可选的文本查询词"}
        }
    def _get_required_args(self) -> List[str]:
        return ["image_urls"]
