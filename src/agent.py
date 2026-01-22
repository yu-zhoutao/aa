import os
import asyncio
from typing import Optional, List, AsyncGenerator
from .llms.openai_llm import OpenAILLM
from .state.state import JudgeState
from .nodes.planning_node import PlanningNode
from .nodes.execution_node import ExecutionNode
from .nodes.report_node import ReportNode
from .utils.config import load_config, Config
from .tools.base import BaseTool
from .utils.sse_utils import SSEUtils

class JudgeAgent:
    def __init__(self, config: Optional[Config] = None, tools: List[BaseTool] = None):
        self.config = config or load_config()
        self.llm = OpenAILLM(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            api_base=self.config.api_url
        )
        
        self.planning_node = PlanningNode(self.llm)
        self.execution_node = ExecutionNode(self.llm, tools or [])
        self.report_node = ReportNode(self.llm)
        
    async def audit(self, file_path: str, file_type: str) -> AsyncGenerator[str, None]:
        """
        执行审核并 yield SSE 事件字符串
        """
        # 定义一个回调，用于在节点内部把事件 yield 出来
        # 由于 Python generator 无法直接从 callback yield，
        # 我们使用 Queue 来解耦
        queue = asyncio.Queue()
        
        async def on_event(event_type: str, content: object):
            sse_msg = SSEUtils.format_event(event_type, content)
            await queue.put(sse_msg)

        # 启动后台任务运行流程
        async def _run_process():
            try:
                state = JudgeState(file_path=file_path, file_type=file_type)
                
                # 1. Plan
                state = await self.planning_node.run(state, on_event)
                
                # 2. Execute
                for i in range(len(state.tasks)):
                    state = await self.execution_node.run_task(state, i, on_event)
                
                # 3. Report
                state = await self.report_node.run(state, on_event)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                await on_event("error", str(e))
            finally:
                # 发送结束信号
                await queue.put(None)

        task = asyncio.create_task(_run_process())

        # 主循环：从队列取消息并 yield
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield msg
            
        await task
