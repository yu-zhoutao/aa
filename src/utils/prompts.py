class PromptTemplates:
    @staticmethod
    def get_image_prompt(labels: str) -> str:
        return f"""请仔细观察图片，判断是否存在以下违规内容：{labels}。
请返回 JSON 格式，包含：
- "image": [违规图片的索引ID列表, 从1开始]
- "reason": "违规原因说明"
如果无违规，"image" 返回空列表 []。
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
        return f"""请审核以下带有时间戳的语音文本内容。
你需要判断其中是否包含违规内容（如色情、暴力、政治敏感、辱骂等）。

文本内容:
{text}

请返回 JSON 格式：
{{
    "is_violation": true/false,
    "time_anchors": [
        {{"start": 10.5, "end": 15.0, "reason": "涉及暴力言论"}}
    ]
}}
如果无违规，"is_violation" 为 false，"time_anchors" 为 []。
"""