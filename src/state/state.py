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
    tool_name: str = ""
    raw_output: Any = None
    finding: str = ""
    is_violation: bool = False
    score: float = 0.0
    evidence_urls: List[str] = field(default_factory=list)
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
    """审核维度任务"""
    dimension: str = ""
    description: str = ""
    status: str = "pending"
    results: List[AuditResult] = field(default_factory=list)
    summary: str = ""
    order: int = 0
    
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
    file_path: str = ""
    file_type: str = ""
    s3_url: str = ""
    
    # 任务清单
    tasks: List[AuditTask] = field(default_factory=list)
    
    # 全局共享上下文 (用于解决重复调用问题)
    # 存储 key -> data，例如 "frames", "ocr_results", "face_results"
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
    final_report: str = ""
    is_violation: bool = False
    is_completed: bool = False
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
            "shared_context": str(self.shared_context)[:500] + "...", # 避免日志过大
            "final_report": self.final_report,
            "is_violation": self.is_violation,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)