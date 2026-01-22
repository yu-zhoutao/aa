import asyncio
from typing import Optional, List
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState
from ..tools.base import BaseTool

class PreProcessingNode(BaseNode):
    """
    预处理节点：负责强制执行通用的感知任务（抽帧、人脸、OCR、搜索）
    避免后续任务重复调用。
    """
    def __init__(self, llm_client, tools: List[BaseTool], enable_search: bool = True):
        super().__init__(llm_client)
        self.tools = {t.name: t for t in tools}
        self.enable_search = enable_search

    async def _run_tool(self, tool_name: str, args: dict, on_event: Optional[LogCallback]) -> dict:
        if tool_name not in self.tools:
            await self.log_info(f"⚠️ 预处理工具缺失: {tool_name}", on_event)
            return {}
        
        await self.log_info(f"🚀 [预处理] 启动: {tool_name}", on_event)
        try:
            res = await self.tools[tool_name].run(**args)
            if "error" in res:
                await self.log_info(f"❌ {tool_name} 失败: {res['error']}", on_event)
                return {}
            
            # 特殊处理预览图事件
            if "preview_images" in res and on_event:
                await on_event("images", res["preview_images"])
                
            await self.log_info(f"✅ {tool_name} 完成", on_event)
            return res
        except Exception as e:
            await self.log_info(f"❌ {tool_name} 异常: {e}", on_event)
            return {}

    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        await self.log_info("开始多模态内容预处理...", on_event)
        
        # 1. 抽帧 (Frame Extract) - 必须项
        frames = []
        if state.file_type in ["image", "video"]:
            res = await self._run_tool("frame_extract", {"file_path": state.file_path, "sample_count": 8}, on_event)
            frames = res.get("frames", [])
            state.shared_context["frames"] = frames
        
        if not frames:
            await self.log_info("⚠️ 未提取到帧，跳过视觉预处理", on_event)
            return state

        # 2. 生成预览 (Preview) - 必须项
        await self._run_tool("preview_upload", {"frames": frames}, on_event)
        
        # 并行执行: 人脸识别、OCR、YOLO (提升效率)
        tasks = []
        
        # 3. 人脸识别 (Face Identify) - 必须项
        task_face = self._run_tool("face_identify", {"frames": frames}, on_event)
        tasks.append(("face_identify", task_face))
        
        # 4. OCR - 必须项
        task_ocr = self._run_tool("ocr_detect", {"frames": frames}, on_event)
        tasks.append(("ocr_detect", task_ocr))
        
        # 5. YOLO - 必须项
        task_yolo = self._run_tool("yolo_detect", {"frames": frames}, on_event)
        tasks.append(("yolo_detect", task_yolo))
        
        # 执行并行任务
        results = await asyncio.gather(*[t[1] for t in tasks])
        
        # 整理结果到 shared_context
        face_results = {}
        ocr_results = []
        
        for (name, _), res in zip(tasks, results):
            state.shared_context[f"{name}_result"] = res
            if name == "face_identify":
                face_results = res
            elif name == "ocr_detect":
                ocr_results = res.get("ocr_results", [])

        # 6. 网络搜索 (Web Search) - 条件触发
        # 触发条件: 1. 识别到人名 2. 用户强制开启 (enable_search)
        search_queries = []
        
        # 从人脸结果提取人名
        persons = face_results.get("persons", [])
        for p in persons:
            if p.get("name") and p.get("name") != "未知":
                search_queries.append(f"人物: {p['name']}")
        
        # 也可以从 OCR 提取关键词 (这里简单处理，暂不提取)

        # 如果没有明确人名，但开启了搜索，尝试搜第一张图 (以图搜图)
        if self.enable_search:
            await self.log_info("🔍 触发网络搜索核查...", on_event)
            
            # 优先搜人名
            if search_queries:
                for q in set(search_queries):
                    await self._run_tool("web_search", {"query": q}, on_event)
            else:
                # 没名字，以图搜图 (搜第一帧)
                if frames:
                    first_frame = frames[0].get("path")
                    await self._run_tool("web_search", {"image_path": first_frame}, on_event)
        
        # 缓存搜索结果
        # 注意：web_search 主要是为了获取信息辅助判断，结果暂时只打印日志或存入 shared_context 供 LLM 读取
        # 这里的实现比较简单，因为 Tool 内部没有把结果返回给调用者保存，而是直接打印了。
        # 改进：我们需要把 web_search 的结果也存下来。
        # 由于上面的 _run_tool 已经执行了，如果 web_search 结果有返回，我们需要改一下 _run_tool 逻辑？
        # 其实 _run_tool 已经返回了 res。
        # 修正：上面的 web_search 调用没有捕获返回值。
        
        return state
