import asyncio
import json
from typing import Optional, List
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState
from ..tools.base import BaseTool
from ..utils.evidence_utils import EvidenceUtils

class PreProcessingNode(BaseNode):
    """
    预处理节点：负责强制执行通用的感知任务
    """
    def __init__(self, llm_client, tools: List[BaseTool], enable_search: bool = True):
        super().__init__(llm_client)
        self.tools = {t.name: t for t in tools}
        self.enable_search = enable_search

    async def _run_tool(self, tool_name: str, args: dict, on_event: Optional[LogCallback]) -> dict:
        if tool_name not in self.tools: return {}
        await self.log_info(f"🚀 [预处理] 启动: {tool_name}", on_event)
        try:
            res = await self.tools[tool_name].run(**args)
            if "error" in res: return {}
            # 只有特定的预览上传事件才发送
            if tool_name == "preview_upload" and "preview_images" in res and on_event:
                await on_event("images", res["preview_images"])
            await self.log_info(f"✅ {tool_name} 完成", on_event)
            return res
        except Exception as e:
            await self.log_info(f"❌ {tool_name} 异常: {e}", on_event)
            return {}

    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        await self.log_info("开始多模态内容预处理...", on_event)
        
        # 1. 抽帧
        frames = []
        if state.file_type in ["image", "video"]:
            res = await self._run_tool("frame_extract", {"file_path": state.file_path, "sample_count": 8}, on_event)
            frames = res.get("frames", [])
            state.shared_context["frames"] = frames
        
        if not frames: return state

        # 2. 生成初步预览 (不带框)
        await self._run_tool("preview_upload", {"frames": frames}, on_event)
        
        # 3. 并行感知 (人脸、OCR、YOLO)
        tasks = []
        tasks.append(("face_identify", self._run_tool("face_identify", {"frames": frames}, on_event)))
        tasks.append(("ocr_detect", self._run_tool("ocr_detect", {"frames": frames}, on_event)))
        tasks.append(("yolo_detect", self._run_tool("yolo_detect", {"frames": frames}, on_event)))
        
        results_list = await asyncio.gather(*[t[1] for t in tasks])
        
        face_res, ocr_detect_res, yolo_res = {}, {}, {}
        for (name, _), res in zip(tasks, results_list):
            state.shared_context[f"{name}_result"] = res
            if name == "face_identify": face_res = res
            elif name == "ocr_detect": ocr_detect_res = res

        # 4. OCR 敏感判定
        ocr_risk_res = {}
        if ocr_detect_res.get("ocr_results"):
            ocr_risk_res = await self._run_tool("ocr_risk_judge", {"ocr_results": ocr_detect_res["ocr_results"]}, on_event)
            state.shared_context["ocr_risk_result"] = ocr_risk_res

        # 5. 汇总证据并统一绘图 (关键：只在这里绘图)
        all_evidence_bboxes = []
        if "evidence_bboxes" in face_res: all_evidence_bboxes.extend(face_res["evidence_bboxes"])
        if "evidence_bboxes" in ocr_risk_res: all_evidence_bboxes.extend(ocr_risk_res["evidence_bboxes"])
            
        if all_evidence_bboxes:
            # 只处理有违规的第一帧（如果是图片），或者按需处理多帧
            # 为简单起见且符合“只保存一张”的需求，我们只处理包含风险的第一帧
            first_risk_frame_idx = all_evidence_bboxes[0]["frame_index"]
            frame_path = next((f["path"] for f in frames if f["index"] == first_risk_frame_idx), None)
            
            if frame_path:
                # 过滤出该帧的所有框
                frame_bboxes = [b for b in all_evidence_bboxes if b["frame_index"] == first_risk_frame_idx]
                evidence_path = EvidenceUtils.generate_evidence_image(frame_path, frame_bboxes)
                if evidence_path:
                    state.shared_context["evidence_images"] = [evidence_path]
                    # 将这张带框证据图也推送给前端
                    if on_event:
                        await on_event("images", [MinioEngine.upload_file(evidence_path)])

        # 6. 搜索情报
        search_findings = []
        if self.enable_search:
            p_names = [p['name'] for p in face_res.get("persons", []) if p.get('name') and p.get('name') != "未知"]
            if p_names:
                for q in set(p_names):
                    s_res = await self._run_tool("web_search", {"query": q}, on_event)
                    if s_res.get("search_findings"): search_findings.append(f"[{q}]: {s_res['search_findings']}")
            elif frames:
                s_res = await self._run_tool("web_search", {"image_path": frames[0]["path"]}, on_event)
                if s_res.get("search_findings"): search_findings.append(f"[搜图]: {s_res['search_findings']}")
        
        state.shared_context["web_search_result"] = search_findings
        return state