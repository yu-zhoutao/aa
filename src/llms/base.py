from abc import ABC, abstractmethod
from typing import Optional

class BaseLLM(ABC):
    def __init__(self, api_key: str, model_name: Optional[str] = None, api_base: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        
    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def ainvoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        pass
