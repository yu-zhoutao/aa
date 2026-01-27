
import re
import json
from typing import Any, List, Dict

class JSONUtils:
    """JSON Parsing Utilities"""

    @staticmethod
    def safe_json_loads(text: str) -> Any:
        if not text:
            return None
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, text)
        if match:
            text = match.group(1)
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                fallback_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if fallback_match:
                    return json.loads(fallback_match.group(1))
            except:
                pass
            print(f"❌ JSON Parse Failed: {text[:100]}...")
            return None

    @staticmethod
    def merge_intervals(intervals: List[Dict[str, Any]], gap: float = 1.0) -> List[Dict[str, Any]]:
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x['start'])
        merged = []
        if not sorted_intervals:
            return merged
        current = sorted_intervals[0].copy()
        for next_int in sorted_intervals[1:]:
            if next_int['start'] <= current['end'] + gap:
                current['end'] = max(current['end'], next_int['end'])
                reasons = set(current.get('reason', '').split('; '))
                reasons.add(next_int.get('reason', ''))
                current['reason'] = "; ".join(filter(None, reasons))
            else:
                merged.append(current)
                current = next_int.copy()
        merged.append(current)
        return merged
