import json
from typing import Optional
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState

SYSTEM_PROMPT_PLANNING = """你是一个多模态内容审核专家。
你的任务是根据用户提供的文件信息（文件类型、名称等），制定一个详细的审核清单。
你需要决定为了确保该文件合法合规，需要从哪些维度进行审核。

【重要原则】
1. 对于视觉内容（图片/视频），**第一步**必须包含"内容预览与识别"，目的是生成预览图并识别画面中的人物。
2. 对于包含人物或敏感标识的内容，必须包含"网络搜索核查"，用于通过搜索引擎确认人物身份或旗帜含义。
3. 必须包含常规的审核维度（色情、暴恐、政治等）。

输出必须是一个 JSON 列表，每个元素包含：
- dimension: 审核维度名称（如：内容预览与识别、政治敏感、色情低俗、网络搜索核查等）
- description: 为什么需要审核这个维度，以及具体的审核重点。

例如：
[
  {"dimension": "内容预览与识别", "description": "提取关键帧并上传生成预览图，同时进行人脸识别以发现敏感人物。"},
  {"dimension": "网络搜索核查", "description": "对识别出的人物或未知标识进行网络搜索，确认其敏感性。"},
  {"dimension": "色情低俗", "description": "检测画面中是否包含裸露、性行为或挑逗性内容。"}
]
"""

class PlanningNode(BaseNode):
    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        await self.log_info(f"正在为文件制定审核计划: {state.file_path}", on_event)
        
        user_prompt = f"文件类型: {state.file_type}\n文件名: {state.file_path}"
        
        response = await self.llm_client.ainvoke(SYSTEM_PROMPT_PLANNING, user_prompt)
        
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3]
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3]
                
            tasks_data = json.loads(clean_response)
            for task in tasks_data:
                state.add_task(task['dimension'], task['description'])
                await self.log_info(f"已添加审核维度: {task['dimension']}", on_event)
        except Exception as e:
            await self.log_info(f"解析审核计划失败: {e}. 使用默认计划。", on_event)
            # 默认计划也加上预览
            state.add_task("内容预览与识别", "生成预览图并识别人脸")
            state.add_task("内容综合审核", "对文件进行全维度的常规合规性检测。")
            
        return state
