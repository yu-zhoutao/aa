
import json
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI, OpenAI
from info_judge_next.config import Config
from info_judge_next.utils.json_utils import JSONUtils

class LLMClient:
    _async_client = None

    @classmethod
    def get_async_client(cls) -> AsyncOpenAI:
        if cls._async_client is None:
            cls._async_client = AsyncOpenAI(
                api_key=Config.VLLM_API_KEY,
                base_url=Config.VLLM_API_URL
            )
        return cls._async_client

    @classmethod
    async def chat_stream(cls, messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        client = cls.get_async_client()
        try:
            stream = await client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_tokens=4096,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield f"\n[LLM Error]: {str(e)}"

    @classmethod
    async def get_json_response(cls, messages: List[Dict[str, Any]]) -> Optional[Dict]:
        client = cls.get_async_client()
        try:
            response = await client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"} if "json" in Config.MODEL_NAME.lower() else None
            )
            content = response.choices[0].message.content
            return JSONUtils.safe_json_loads(content)
        except Exception as e:
            print(f"❌ LLM JSON Error: {e}")
            return None
