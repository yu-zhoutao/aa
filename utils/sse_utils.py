
import json
import time
from typing import Any, Optional

class SSEUtils:
    """SSE Format Utilities"""

    @staticmethod
    def format_event(event_type: str, content: Any) -> str:
        data = {
            "type": event_type,
            "content": content
        }
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n"

    @staticmethod
    def log(message: str, start_time: Optional[float] = None) -> str:
        if start_time is not None:
            elapsed = time.time() - start_time
            message = f"[{elapsed:.1f}s] {message}"
        
        return SSEUtils.format_event("log", message)

    @staticmethod
    def error(message: str) -> str:
        return SSEUtils.format_event("error", message)

    @staticmethod
    def token(content: str) -> str:
        return SSEUtils.format_event("token", content)

    @staticmethod
    def images(image_list: list) -> str:
        return SSEUtils.format_event("images", image_list)

    @staticmethod
    def violation(data: dict) -> str:
        return SSEUtils.format_event("violation_data", data)
