from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, Optional
from ..llms.base import BaseLLM
from ..state.state import JudgeState

# 定义回调函数类型
LogCallback = Callable[[str, str], Awaitable[None]] # func(event_type, content) -> None

class BaseNode(ABC):
    def __init__(self, llm_client: BaseLLM, node_name: str = ""):
        self.llm_client = llm_client
        self.node_name = node_name or self.__class__.__name__

    @abstractmethod
    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        pass

    async def emit(self, on_event: Optional[LogCallback], event_type: str, content: Any):
        """发送事件"""
        if on_event:
            await on_event(event_type, content)

    async def log_info(self, message: str, on_event: Optional[LogCallback] = None):
        """记录信息日志并发送 SSE 事件"""
        print(f"[{self.node_name}] {message}")
        if on_event:
            await on_event("log", f"[{self.node_name}] {message}")