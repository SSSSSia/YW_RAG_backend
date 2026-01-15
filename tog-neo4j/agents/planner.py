"""
Agent 任务规划器
"""
import re
from typing import Dict, Any, List, Optional
from agents.base import AgentContext, AgentResult
from agents.registry import agent_registry
from core.llm_client import llm_client
from utils.logger import logger


class AgentPlanner:
    """Agent 任务规划器"""

    def __init__(self):
        self.planning_prompt_template = """你是一个知识图谱查询系统的任务规划助手。请分析用户的请求，选择最合适的处理方式。

可用的 Agent 及其能力：
{capabilities}

用户请求：{query}

grag_id：{grag_id}

请分析用户的意图，并选择最合适的 Agent。如果用户明确指定了查询方式，请遵循用户的选择。

请只返回 Agent 名称，格式如下：
AGENT: <agent_name>

如果需要额外参数，可以添加：
PARAMS: key1=value1, key2=value2
"""

    async def plan(self, context: AgentContext) -> Optional[str]:
        """规划任务，返回推荐的 Agent 名称"""
        try:
            # 获取所有 Agent 的能力
            capabilities = agent_registry.get_all_capabilities()
            print( capabilities)
            # 格式化能力描述
            caps_text = self._format_capabilities(capabilities)

            # 构建规划提示
            prompt = self.planning_prompt_template.format(
                capabilities=caps_text,
                query=context.question,
                grag_id=context.grag_id
            )

            # 调用 LLM 进行规划
            response = await llm_client.generate(prompt, temperature=0.1)

            # 解析响应
            agent_name = self._parse_agent_name(response)
            if agent_name:
                logger.info(f"📋 任务规划完成，推荐使用: {agent_name}")

                # 解析额外参数
                params = self._parse_params(response)
                if params:
                    context.metadata.update(params)

                return agent_name
            else:
                logger.warning("⚠️ 无法从规划结果中提取 Agent 名称")
                return None

        except Exception as e:
            logger.error(f"❌ 任务规划失败: {e}", exc_info=True)
            return None

    def _format_capabilities(self, capabilities: Dict[str, Dict]) -> str:
        """格式化 Agent 能力描述"""
        lines = []
        for name, cap in capabilities.items():
            lines.append(f"- {name}: {cap['description']}")
            if cap.get('tools'):
                lines.append(f"  工具: {', '.join(cap['tools'])}")
        return "\n".join(lines)

    def _parse_agent_name(self, response: str) -> Optional[str]:
        """从响应中解析 Agent 名称"""
        match = re.search(r'AGENT:\s*(\w+)', response, re.IGNORECASE)
        if match:
            return match.group(1)

        # 尝试直接匹配已知的 Agent 名称
        known_agents = agent_registry.list_agents()
        for agent in known_agents:
            if agent.lower() in response.lower():
                return agent

        return None

    def _parse_params(self, response: str) -> Dict[str, Any]:
        """从响应中解析参数"""
        params = {}
        match = re.search(r'PARAMS:\s*(.+)', response, re.IGNORECASE)
        if match:
            param_str = match.group(1)
            for item in param_str.split(','):
                if '=' in item:
                    key, value = item.split('=', 1)
                    params[key.strip()] = value.strip()
        return params


# 全局规划器实例
agent_planner = AgentPlanner()