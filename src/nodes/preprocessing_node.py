import asyncio
import json
from typing import Optional, List
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState
from ..tools.base import BaseTool
from ..utils.evidence_utils import EvidenceUtils
from ..engines.minio_engine import MinioEngine

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
            
            # 修改：不再主动推送 preview_upload 的原图
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

        # 2. 生成初步预览 (仅内部使用，不推送到前端)
        # 传入 None 作为 on_event，防止内部 log 刷屏，或者继续传 on_event 但 _run_tool 已屏蔽图片
        preview_res = await self._run_tool("preview_upload", {"frames": frames}, on_event)
        current_preview_images = preview_res.get("preview_images", [])
        
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

        # 5. 汇总证据并统一绘图
        all_evidence_bboxes = []
        if "evidence_bboxes" in face_res: all_evidence_bboxes.extend(face_res["evidence_bboxes"])
        if "evidence_bboxes" in ocr_risk_res: all_evidence_bboxes.extend(ocr_risk_res["evidence_bboxes"])
            
        if all_evidence_bboxes:
            # 按帧分组 bbox
            bboxes_by_frame = {}
            for bbox in all_evidence_bboxes:
                f_idx = bbox.get("frame_index", 0)
                if f_idx not in bboxes_by_frame: bboxes_by_frame[f_idx] = []
                bboxes_by_frame[f_idx].append(bbox)
            
            # 遍历每一帧，如果有风险标记，则重新生成图片
            for list_idx, frame_item in enumerate(frames):
                f_idx = frame_item["index"]
                if f_idx in bboxes_by_frame:
                    bboxes = bboxes_by_frame[f_idx]
                    frame_path = frame_item.get("path")
                    if frame_path:
                        evidence_path = EvidenceUtils.generate_evidence_image(frame_path, bboxes)
                        if evidence_path:
                            try:
                                marked_url = MinioEngine.upload_file(evidence_path)
                                # 使用 list_idx 更新预览列表中的对应位置，而不是使用 f_idx (视频帧号)
                                if list_idx < len(current_preview_images):
                                    original_url = current_preview_images[list_idx]
                                    print(f"🔄 [PreProcessing] 替换第 {list_idx} 帧: {original_url} -> {marked_url}")
                                    if original_url == marked_url:
                                        print(f"⚠️ [PreProcessing] 警告: 标记后的图片 URL 与原图相同，MinIO 可能判定内容一致或去重异常。")
                                    current_preview_images[list_idx] = marked_url
                                else:
                                    print(f"⚠️ [PreProcessing] 无法替换图片: list_idx {list_idx} 超出 preview_images 长度 {len(current_preview_images)}")
                            except Exception as e:
                                print(f"❌ 证据图上传失败: {e}")

        # 无论是否有更新，统一在这里推送最终确认的图片列表
        if on_event:
            print(f"📤 [PreProcessing] 推送最终图片列表 (Count: {len(current_preview_images)})")
            await on_event("images", current_preview_images)

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
