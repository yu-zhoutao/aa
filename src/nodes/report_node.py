import json
from typing import Optional
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState

SYSTEM_PROMPT_REPORT = """你是一个内容审核报告生成器。
你需要根据所有的审核任务结果，生成一份最终的判定报告。
判定结论必须明确：[违规] 或 [合规]。

报告应包含：
1. 审核概览
2. 各维度详细发现
3. 最终判定结论与理由

数据:
{data}
"""

class ReportNode(BaseNode):
    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        await self.log_info("正在生成最终报告...", on_event)
        
        tasks_data = [t.to_dict() for t in state.tasks]
        data_str = json.dumps(tasks_data, ensure_ascii=False, indent=2)
        
        if on_event:
            await on_event("final_report_start", "")

        # TODO: 这里应该支持 stream 输出
        response = await self.llm_client.ainvoke(SYSTEM_PROMPT_REPORT, data_str)
        
        # 模拟流式输出
        if on_event:
            for char in response:
                await on_event("token", char)
                # await asyncio.sleep(0.002) 

            await on_event("final_report_end", "")

        state.final_report = response
        state.is_violation = "[违规]" in response
        state.is_completed = True
        
        return state