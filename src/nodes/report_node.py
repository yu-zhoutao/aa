import json
from typing import Optional
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState

SYSTEM_PROMPT_REPORT = """你是一个专业的内容安全审核报告生成器。
你需要根据所有的审核任务结果，生成一份结构化、详细的最终判定报告。

报告格式要求：
1. **多模态内容摘要**：
   - 根据 OCR 结果、人脸识别结果和画面描述，总结文件里到底有什么（例如：图片中显示了一群人在...，其中包含了...文字）。
   - 如果有人脸识别结果，必须明确指出识别到了谁。
   - 如果有 OCR 结果，简要概括主要文字内容。

2. **各维度详细发现**：
   - 逐一列出每个审核维度的发现。
   - 重点描述发现的风险点。

3. **最终判定结论**：
   - 明确给出：[违规] 或 [合规] 或 [疑似违规]。
   - 给出判定理由。

数据输入:
{data}
"""

class ReportNode(BaseNode):
    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        await self.log_info("正在生成最终报告...", on_event)
        
        # 准备数据，可能比较大，注意 Token 限制
        # 这里可以优化：只提取 finding 和 is_violation 字段
        tasks_data = []
        for t in state.tasks:
            task_summary = {
                "dimension": t.dimension,
                "results": []
            }
            for r in t.results:
                # 过滤掉冗余的 raw_output，只保留 finding 和关键信息
                result_lite = {
                    "tool": r.tool_name,
                    "finding": r.finding,
                    "is_violation": r.is_violation
                }
                # 特殊处理：如果是人脸识别或OCR，保留关键数据
                if r.tool_name == "face_identify" and isinstance(r.raw_output, dict):
                    result_lite["persons"] = r.raw_output.get("persons", [])
                if r.tool_name == "ocr_detect" and isinstance(r.raw_output, dict):
                     # OCR 结果可能很长，只取前 200 字
                     ocr_items = r.raw_output.get("ocr_results", [])
                     texts = [item.get("text", "") for sub in ocr_items for item in sub.get("items", [])]
                     result_lite["ocr_sample"] = "".join(texts)[:200]
                
                task_summary["results"].append(result_lite)
            tasks_data.append(task_summary)

        data_str = json.dumps(tasks_data, ensure_ascii=False, indent=2)
        
        if on_event:
            await on_event("final_report_start", "")

        response = await self.llm_client.ainvoke(SYSTEM_PROMPT_REPORT, data_str)
        
        if on_event:
            # 模拟流式输出
            chunk_size = 5
            for i in range(0, len(response), chunk_size):
                await on_event("token", response[i:i+chunk_size])
                await asyncio.sleep(0.01)

            await on_event("final_report_end", "")

        state.final_report = response
        state.is_violation = "[违规]" in response
        state.is_completed = True
        
        return state
