import json
from typing import List, Dict, Optional
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState, AuditResult
from ..tools.base import BaseTool

SYSTEM_PROMPT_EXECUTION = """你是一个任务分配器。
你有一个任务维度描述，和一系列可用的工具。
你需要决定使用哪些工具来完成这个审核维度的任务。

审核维度: {dimension}
任务描述: {description}

可用工具:
{tools_info}

输出必须是一个 JSON 列表，包含你决定调用的工具名称及其参数。
例如: [{"tool_name": "frame_extract", "args": {"sample_count": 10}}]
"""

class ExecutionNode(BaseNode):
    def __init__(self, llm_client, tools: List[BaseTool]):
        super().__init__(llm_client)
        self.tools = {t.name: t for t in tools}
        self.tools_info = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    async def run_task(self, state: JudgeState, task_index: int, on_event: Optional[LogCallback] = None) -> JudgeState:
        task = state.tasks[task_index]
        await self.log_info(f"执行审核维度: {task.dimension}", on_event)
        task.status = "running"
        
        user_prompt = f"维度: {task.dimension}\n描述: {task.description}\n文件路径: {state.file_path}"
        sys_prompt = SYSTEM_PROMPT_EXECUTION.format(
            dimension=task.dimension, 
            description=task.description,
            tools_info=self.tools_info
        )
        
        response = await self.llm_client.ainvoke(sys_prompt, user_prompt)
        
        try:
            clean_response = response.strip()
            if "```" in clean_response:
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            
            calls = json.loads(clean_response)
            
            for call in calls:
                tool_name = call['tool_name']
                args = call.get('args', {})
                if 'file_path' not in args:
                    args['file_path'] = state.file_path
                
                if tool_name in self.tools:
                    await self.log_info(f"🚀 调用工具: {tool_name}", on_event)
                    
                    # 执行工具
                    tool_result = await self.tools[tool_name].run(**args)
                    
                    # 发送特定事件 (如图片预览)
                    if "preview_images" in tool_result and on_event:
                        await on_event("images", tool_result["preview_images"])
                    
                    # 记录结果
                    result_obj = AuditResult(
                        tool_name=tool_name,
                        raw_output=tool_result,
                        finding=f"工具 {tool_name} 执行完成。"
                    )
                    task.add_result(result_obj)
                    await self.log_info(f"✅ 工具 {tool_name} 完成", on_event)
                else:
                    await self.log_info(f"❌ 找不到工具: {tool_name}", on_event)
                    
            task.status = "completed"
        except Exception as e:
            await self.log_info(f"❌ 执行任务失败: {e}", on_event)
            task.status = "failed"
            
        return state

    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        return state