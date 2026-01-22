"""
Judge Agent 状态管理
定义审核过程中的所有状态数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


@dataclass
class AuditResult:
    """单个审计工具执行的结果"""
    tool_name: str = ""                # 使用的工具名称
    raw_output: Any = None             # 工具原始输出
    finding: str = ""                  # LLM对结果的理解/发现
    is_violation: bool = False         # 是否发现违规
    score: float = 0.0                 # 违规严重程度评分 (0-1)
    evidence_urls: List[str] = field(default_factory=list) # 证据图片/视频切片URL
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "raw_output": self.raw_output,
            "finding": self.finding,
            "is_violation": self.is_violation,
            "score": self.score,
            "evidence_urls": self.evidence_urls,
            "timestamp": self.timestamp
        }


@dataclass
class AuditTask:
    """审核维度任务（类似 DeepSearchAgent 的 Paragraph）"""
    dimension: str = ""                # 审核维度 (如: 政治敏感, 色情低俗, 语音违规)
    description: str = ""              # 任务描述
    status: str = "pending"            # pending, running, completed, failed
    results: List[AuditResult] = field(default_factory=list) # 该维度下的多次检测结果
    summary: str = ""                  # 该维度的最终审核小结
    order: int = 0                     # 任务顺序
    
    def is_completed(self) -> bool:
        return self.status == "completed"
    
    def add_result(self, result: AuditResult):
        self.results.append(result)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "description": self.description,
            "status": self.status,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "order": self.order
        }


@dataclass
class JudgeState:
    """整个审核流程的状态"""
    file_path: str = ""                # 待审核文件本地路径
    file_type: str = ""                # 文件类型 (video, image, audio)
    s3_url: str = ""                   # 远程存储URL
    tasks: List[AuditTask] = field(default_factory=list) # 审核任务清单
    final_report: str = ""             # 最终判定报告
    is_violation: bool = False         # 整体判定是否违规
    is_completed: bool = False         # 审核流程是否结束
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_task(self, dimension: str, description: str) -> int:
        order = len(self.tasks)
        task = AuditTask(dimension=dimension, description=description, order=order)
        self.tasks.append(task)
        self.update_timestamp()
        return order
    
    def update_timestamp(self):
        self.updated_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "s3_url": self.s3_url,
            "tasks": [t.to_dict() for t in self.tasks],
            "final_report": self.final_report,
            "is_violation": self.is_violation,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
            
    @classmethod
    def load_from_file(cls, filepath: str) -> "JudgeState":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 这里需要复杂的解析逻辑来还原嵌套对象，目前简化处理
        return cls(**data)
