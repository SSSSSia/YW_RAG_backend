"""
Agent 基类定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class AgentStatus(Enum):
    """Agent 执行状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class AgentContext:
    """Agent 执行上下文"""
    grag_id: str
    question: str
    conversation_history: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    data: Any
    message: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.tools = []

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """执行 Agent 任务"""
        pass

    @abstractmethod
    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理该任务"""
        pass

    def register_tool(self, tool):
        """注册工具"""
        self.tools.append(tool)

    def get_capabilities(self) -> Dict[str, Any]:
        """获取 Agent 能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "tools": [tool.name for tool in self.tools]
        }