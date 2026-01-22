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
