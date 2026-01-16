"""
Agent 模块
"""
from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.registry import agent_registry, AgentRegistry
from agents.planner import agent_planner, AgentPlanner
from agents.impl import AutoQueryAgent
from agents.impl import LangChainQueryAgent
from agents.impl import LangGraphAgent


# 初始化并注册所有 Agent
def initialize_agents():
    """初始化并注册所有 Agent"""

    agent_registry.register(AutoQueryAgent())
    agent_registry.register(LangChainQueryAgent())
    agent_registry.register(LangGraphAgent())


__all__ = [
    'BaseAgent',
    'AgentContext',
    'AgentResult',
    'AgentStatus',
    'agent_registry',
    'AgentRegistry',
    'agent_planner',
    'AgentPlanner',
    'AutoQueryAgent',
    'initialize_agents'
]