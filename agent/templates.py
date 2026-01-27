from typing import List, Union

class PromptTemplates:
    """Standardized LLM Prompt Templates"""

    @staticmethod
    def audio_correction_prompt(text: str) -> str:
        """Basic correction for transcribed text"""
        return f"对语音转写文本纠错，修正同音字和标点，直接输出原文：：\n{text}"

    @staticmethod
    def text_review_and_correct_json_template(text_lines: Union[str, List[str]]) -> str:
        """
        Violation detection and timestamp extraction for transcribed text (JSON format)
        """
        return f"""你是一个内容安全专家。下面是语音转写的文本内容（带有时间戳）：
{text_lines}

请分析上述内容，识别出其中涉及“政治敏感”、“违规标识”、“违法违规”、“同性”、“低俗”、“擦边卖腐”的片段。
你需要返回一个 JSON 对象，格式如下：
{{
  "is_violation": true/false,
  "time_anchors": [
    {{
      "start": 开始时间(float),
      "end": 结束时间(float),
      "reason": "违规原因简述"
    }}
  ]
}}
注意：
1. 如果没有违规，is_violation 为 false，time_anchors 为空列表。
2. 只要返回 JSON，不要任何其他解释。"""
