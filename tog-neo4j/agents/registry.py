"""
Agent 注册中心
"""
from typing import Dict, List, Optional
from agents.base import BaseAgent, AgentContext
from utils.logger import logger


class AgentRegistry:
    """Agent 注册中心"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        """注册 Agent"""
        if agent.name in self._agents:
            logger.warning(f"Agent '{agent.name}' 已存在，将被覆盖")

        self._agents[agent.name] = agent
        # logger.info(f"✅ Agent '{agent.name}' 注册成功")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """根据名称获取 Agent"""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """列出所有已注册的 Agent"""
        return list(self._agents.keys())

    def find_suitable_agent(self, context: AgentContext) -> Optional[BaseAgent]:
        """查找合适的 Agent 处理任务"""
        for agent in self._agents.values():
            if agent.can_handle(context):
                logger.info(f"🎯 找到合适的 Agent: {agent.name}")
                return agent

        logger.warning("⚠️ 未找到合适的 Agent")
        return None

    def get_all_capabilities(self) -> Dict[str, Dict]:
        """获取所有 Agent 的能力描述"""
        return {
            name: agent.get_capabilities()
            for name, agent in self._agents.items()
        }


# 全局 Agent 注册中心实例
agent_registry = AgentRegistry()