
from typing import Dict, List, Any, Union

class BaseTool:
    name: str = ""
    description: str = ""

    async def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def to_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self._get_args_schema(),
                    "required": self._get_required_args()
                }
            }
        }

    def _get_args_schema(self) -> Dict:
        return {}

    def _get_required_args(self) -> List[str]:
        return []
