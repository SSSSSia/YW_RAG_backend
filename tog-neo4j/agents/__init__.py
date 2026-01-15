"""
Agent 模块
"""
from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.registry import agent_registry, AgentRegistry
from agents.planner import agent_planner, AgentPlanner
from agents.impl import AutoQueryAgent


# 初始化并注册所有 Agent
def initialize_agents():
    """初始化并注册所有 Agent"""
    auto_query_agent = AutoQueryAgent()
    agent_registry.register(auto_query_agent)


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