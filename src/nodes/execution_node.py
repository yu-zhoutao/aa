import json
from typing import List, Dict, Optional
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState, AuditResult
from ..tools.base import BaseTool
from ..utils.json_utils import JSONUtils

SYSTEM_PROMPT_EXECUTION = """你是一个智能审核执行器。
你有一个审核任务，以及一些**已经执行过的预处理结果**（如OCR文字、人脸识别结果）。
你需要利用这些已有信息，或者调用特定的判定工具（如 behavior_judge），来完成当前维度的审核。

审核维度: {dimension}
任务描述: {description}

【已有的预处理数据】
{context_summary}

【可用工具】
{tools_info}

【指令】
1. 如果已有的数据足以判断（例如：OCR已经识别出了敏感词），则不需要再调用OCR工具，直接使用数据判定即可。
2. 针对由于缺乏语义理解而无法直接判定的维度（如色情、政治隐喻），请积极调用 `behavior_judge` 工具。
3. 请只返回你需要**新调用**的工具。如果不需要调用任何工具就能下结论，请返回空列表 []。

输出 JSON 格式:
例如: [{{"tool_name": "behavior_judge", "args": {{}}}}]"""

class ExecutionNode(BaseNode):
    def __init__(self, llm_client, tools: List[BaseTool]):
        super().__init__(llm_client)
        self.tools = {t.name: t for t in tools}
        self.tools_info = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    def _get_context_summary(self, state: JudgeState) -> str:
        summary = []
        ctx = state.shared_context
        
        if "face_identify_result" in ctx:
            persons = ctx["face_identify_result"].get("persons", [])
            names = [p["name"] for p in persons if p.get("name")]
            summary.append(f"- 人脸识别: 发现 {len(persons)} 人, 姓名: {', '.join(names) or '无'}")
            
        if "ocr_detect_result" in ctx:
            ocr_res = ctx["ocr_detect_result"].get("ocr_results", [])
            total_chars = sum(len(item.get("text", "")) for sub in ocr_res for item in sub.get("items", []))
            summary.append(f"- OCR识别: 已执行，共识别约 {total_chars} 字符")
            
        if "yolo_detect_result" in ctx:
            yolo_res = ctx["yolo_detect_result"].get("detections", [])
            labels = set()
            for sub in yolo_res:
                for det in sub.get("bboxes", []):
                    labels.add(det.get("label", "obj") )
            summary.append(f"- 物体检测: 发现类别 {list(labels)}")
            
        return "\n".join(summary) or "无预处理数据"

    async def run_task(self, state: JudgeState, task_index: int, on_event: Optional[LogCallback] = None) -> JudgeState:
        task = state.tasks[task_index]
        await self.log_info(f"执行审核维度: {task.dimension}", on_event)
        task.status = "running"
        
        frames = state.shared_context.get("frames", [])
        
        user_prompt = f"维度: {task.dimension}\n描述: {task.description}\n文件路径: {state.file_path}"
        context_summary = self._get_context_summary(state)
        
        sys_prompt = SYSTEM_PROMPT_EXECUTION.format(
            dimension=task.dimension, 
            description=task.description,
            context_summary=context_summary,
            tools_info=self.tools_info
        )
        
        response = await self.llm_client.ainvoke(sys_prompt, user_prompt)
        
        print(f"\n   [模型决策: {task.dimension}]")
        print(f"   {response}")
        
        try:
            calls = JSONUtils.safe_json_loads(response)
            
            # 兼容单个对象的情况
            if isinstance(calls, dict):
                calls = [calls]
            elif not isinstance(calls, list):
                calls = [] # 解析失败或为空
            
            for call in calls:
                tool_name = call.get('tool_name')
                if not tool_name: continue
                
                args = call.get('args', {})
                
                if 'file_path' not in args:
                    args['file_path'] = state.file_path
                if 'frames' not in args and frames:
                    args['frames'] = frames
                
                if tool_name in self.tools:
                    await self.log_info(f"🚀 [针对性复查] 调用: {tool_name}", on_event)
                    tool_result = await self.tools[tool_name].run(**args)
                    
                    # --- 新增：向前端推送多模态证据 ---
                    if on_event:
                        # 1. 图片预览推送
                        if "preview_images" in tool_result:
                            await on_event("images", tool_result["preview_images"])
                        
                        # 2. 违规切片数据推送
                        if "violation_check" in tool_result:
                            v_data = tool_result["violation_check"]
                            if v_data.get("is_violation"):
                                frontend_data = {
                                    "is_violation": True,
                                    "time_anchors": v_data.get("segments", [])
                                }
                                await on_event("violation_data", frontend_data)
                    
                    finding = f"工具 {tool_name} 执行完成。"
                    is_violation = False
                    
                    if tool_name == "behavior_judge":
                         risks = tool_result.get("visual_risks", [])
                         if risks:
                             finding = f"发现风险: {'; '.join(risks)}"
                             is_violation = True
                    
                    result_obj = AuditResult(
                        tool_name=tool_name,
                        raw_output=tool_result,
                        finding=finding,
                        is_violation=is_violation
                    )
                    task.add_result(result_obj)
                    
                    print(f"   -> [{tool_name}] 违规: {is_violation}, 发现: {finding[:50]}...")
                    
                else:
                    await self.log_info(f"❌ 找不到工具: {tool_name}", on_event)
                    
            task.status = "completed"
        except Exception as e:
            await self.log_info(f"❌ 执行任务失败: {e}", on_event)
            task.status = "failed"
        
        print(f"   [维度结束] {task.dimension}: 完成")
        print("-" * 30)
            
        return state

    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        return state
