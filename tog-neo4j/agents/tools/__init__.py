"""
Agents 模块初始化
"""
from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.registry import agent_registry
from agents.planner import agent_planner

# 导入新的 LangChain Agent
from agents.impl.langchain_agent import LangChainQueryAgent
from agents.impl.langgraph_agent import LangGraphAgent

from agents.tools.query_tog import QueryToGTool
from agents.tools.query_graphrag import QueryGraphRAGTool
from agents.tools.query_hybrid import QueryHybridTool

from utils.logger import logger


def initialize_agents():
    """初始化所有 Agent"""
    logger.info("🚀 开始初始化 Agent 系统...")

    # 注册原有 Agent (保持兼容性)
    agent_registry.register(AutoQueryAgent())

    # 注册新的 LangChain Agent
    agent_registry.register(LangChainQueryAgent())

    # 注册 LangGraph Agent (推荐使用)
    agent_registry.register(LangGraphAgent())

    logger.info(f"✅ 共注册 {len(agent_registry.list_agents())} 个 Agent")


__all__ = [
    # 基础类
    'BaseAgent',
    'AgentContext',
    'AgentResult',
    'AgentStatus',

    # 注册中心和规划器
    'agent_registry',
    'agent_planner',

    # Agent 实现
    'AutoQueryAgent',
    'LangChainQueryAgent',
    'LangGraphAgent',

    'QueryToGTool',
    'QueryGraphRAGTool',
    'QueryHybridTool',

    # 初始化函数
    'initialize_agents'

]