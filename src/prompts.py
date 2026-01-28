class PromptTemplates:
    @staticmethod
    def get_image_prompt(bad_type: str) -> str:
        """
        用于通用视觉违规行为判定的提示词模板
        """
        return f"""你是一位专业的内容安全审核专家。
任务：请分析以下提供的图片序列中，哪些内容涉及【{bad_type}】。

【判定维度】：
1. 违规行为：是否存在攻击性手势、非法集会、危险动作或违背社会公德的行为。
2. 敏感标识：是否包含违规旗帜、邪教符号、受限组织的徽标或水印。
3. 丑化阴暗：是否涉及嘲讽/丑化公众人物、展现极端人性阴暗面或血腥不适的内容。

【输出要求】：
- 必须使用 JSON 格式输出。
- 格式示例：{{"image": [1, 3]}}  (数字对应图片的顺序编号，从1开始)
- 如果全部合规，输出：{{"image": []}}
- 严禁输出任何解释、分析或多余文字。

"""

    @staticmethod
    def ocr_judge_prompt(text_full: str, text_map: dict) -> str:
        return f"""请审核以下 OCR 识别出的文本内容：
全文: {text_full}

请判断是否存在敏感、违禁、色情、暴力或政治不当内容。
如果存在，请返回 JSON:
{{
    "id": [违规文本段落的ID列表],
    "reason": "违规原因"
}}
如果无违规，"id" 返回 []。

文本段落映射:
{text_map}
"""

    @staticmethod
    def audio_correction_prompt(text: str) -> str:
        return f"""以下是一段语音转写的文本，可能包含识别错误或语气词。
请对其进行纠错和润色，使其更加通顺，去除无意义的语气词，但保留原意。
直接返回修改后的文本，不要包含额外解释。

原文:
{text}
"""

    @staticmethod
    def text_review_and_correct_json_template(text: str) -> str:
        return f"""你是一个内容安全专家。下面是语音转写的文本内容（带有时间戳）：
你需要判断其中是否包含违规内容（如色情、暴力、政治敏感、辱骂等）。

请分析下述内容，识别出其中涉及“政治敏感”、“违规标识”、“违法违规”、“同性”、"低俗"、"擦边卖腐"的片段。
{text}

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
1.如果无违规，"is_violation" 为 false，"time_anchors" 为 []。
2. 只要返回 JSON，不要任何其他解释。
"""