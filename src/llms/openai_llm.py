import openai
from typing import Optional, List, Dict, Any
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str, api_base: Optional[str] = None):
        super().__init__(api_key, model_name, api_base)
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)

    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    async def ainvoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
