import asyncio
import json
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
        
        # 1. 抽帧 (Frame Extract)
        frames = []
        if state.file_type in ["image", "video"]:
            res = await self._run_tool("frame_extract", {"file_path": state.file_path, "sample_count": 8}, on_event)
            frames = res.get("frames", [])
            state.shared_context["frames"] = frames
        
        if not frames:
            await self.log_info("⚠️ 未提取到帧，跳过视觉预处理", on_event)
            return state

        # 2. 生成预览 (Preview)
        await self._run_tool("preview_upload", {"frames": frames}, on_event)
        
        # 并行执行: 人脸识别、OCR、YOLO
        tasks = []
        task_face = self._run_tool("face_identify", {"frames": frames}, on_event)
        tasks.append(("face_identify", task_face))
        
        task_ocr = self._run_tool("ocr_detect", {"frames": frames}, on_event)
        tasks.append(("ocr_detect", task_ocr))
        
        task_yolo = self._run_tool("yolo_detect", {"frames": frames}, on_event)
        tasks.append(("yolo_detect", task_yolo))
        
        results = await asyncio.gather(*[t[1] for t in tasks])
        
        # 整理结果到 shared_context
        face_results = {}
        for (name, _), res in zip(tasks, results):
            state.shared_context[f"{name}_result"] = res
            if name == "face_identify":
                face_results = res

        # 6. 网络搜索 (Web Search)
        search_queries = []
        persons = face_results.get("persons", [])
        for p in persons:
            if p.get("name") and p.get("name") != "未知":
                search_queries.append(f"人物: {p['name']}")
        
        search_findings = []
        if self.enable_search:
            await self.log_info("🔍 触发网络搜索核查...", on_event)
            if search_queries:
                for q in set(search_queries):
                    s_res = await self._run_tool("web_search", {"query": q}, on_event)
                    if s_res.get("search_findings"):
                        search_findings.append(f"[{q}]: {s_res['search_findings']}")
            else:
                if frames:
                    first_frame = frames[0].get("path")
                    s_res = await self._run_tool("web_search", {"image_path": first_frame}, on_event)
                    if s_res.get("search_findings"):
                        search_findings.append(f"[以图搜图]: {s_res['search_findings']}")
        
        # 保存搜索结果到共享上下文
        state.shared_context["web_search_result"] = search_findings
        
        # === 打印预处理摘要 ===
        print("\n" + "="*50)
        print("📊 预处理结果摘要:")
        
        p_names = [p['name'] for p in persons if p.get('name')]
        print(f"   - 人脸识别: {len(persons)} 人, 姓名: {p_names}")
        
        ocr_text = ""
        if "ocr_detect_result" in state.shared_context:
            for item in state.shared_context["ocr_detect_result"].get("ocr_results", []):
                for sub in item.get("items", []):
                    ocr_text += sub.get("text", "")
        print(f"   - OCR文字: {len(ocr_text)} 字, 预览: {ocr_text[:50]}...")
        
        labels = set()
        if "yolo_detect_result" in state.shared_context:
            for item in state.shared_context["yolo_detect_result"].get("detections", []):
                for b in item.get("bboxes", []):
                    labels.add(b.get("label"))
        print(f"   - 目标检测: {list(labels)}")
        
        if search_findings:
            print(f"   - 网络搜索: 获得 {len(search_findings)} 条情报")
        print("="*50 + "\n")
        
        return state
