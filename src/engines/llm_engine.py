import base64
from typing import List, Dict, Any, Union
from ..llms.openai_llm import OpenAILLM
from ..config import Config

class LLMEngine:
    """为工具提供的 LLM 调用接口"""
    
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            config = Config()
            cls._client = OpenAILLM(
                api_key=config.api_key,
                model_name=config.model_name,
                api_base=config.api_url
            )
        return cls._client

    @classmethod
    async def get_json_response(cls, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取 JSON 格式的响应
        """
        client = cls.get_client()
        # 提取 system prompt 和 user prompt
        system_prompt = "You are a helpful assistant."
        user_prompt = ""
        
        # 简单的消息合并逻辑 (适配 BaseLLM 接口)
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                if isinstance(msg['content'], str):
                    user_prompt += msg['content'] + "\n"
                elif isinstance(msg['content'], list):
                    # 处理多模态消息 (BaseLLM 目前可能不支持，需要检查)
                    # 如果 BaseLLM 是纯文本的，这里会丢失图片信息
                    # 但 OpenAILLM 类实际上使用的是 openai SDK，支持多模态
                    pass

        # 直接调用 openai SDK 以支持多模态
        try:
            response = await client.async_client.chat.completions.create(
                model=client.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            import json
            return json.loads(content)
        except Exception as e:
            print(f"LLM JSON 请求失败: {e}")
            return {{}}

    @staticmethod
    def build_visual_message(prompt: str, images_b64: List[str]) -> List[Dict[str, Any]]:
        """构建多模态消息"""
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        return [{"role": "user", "content": content}]
