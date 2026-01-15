"""
Agent 服务 - 统一的 Agent 调度和管理
"""
import time
from typing import Dict, Any, List, Optional
from agents import (
    AgentContext, AgentResult, agent_registry,
    agent_planner, initialize_agents
)
from utils.java_backend import get_knowledge_bases
from utils.logger import logger


class AgentService:
    """Agent 服务"""

    def __init__(self):
        # 初始化所有 Agent
        initialize_agents()
        logger.info("✅ Agent 系统初始化完成")

    async def process_request(
            self,
            grag_id: str,
            question: str,
            conversation_history: List[Dict[str, str]] = None,
            metadata: Dict[str, Any] = None,
            agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户请求

        Args:
            grag_id: 图谱ID
            question: 用户查询
            conversation_history: 对话历史
            metadata: 额外元数据
            agent_name: 指定的Agent名称（可选）

        Returns:
            处理结果
        """
        start_time = time.time()

        try:
            logger.info("=" * 60)
            logger.info(f"[{grag_id}] 🤖 Agent服务收到请求")
            logger.info(f"查询: {question}")

            # 【新增】如果提供了 company_id，从 Java 后端获取知识库列表
            company_id = (metadata or {}).get("company_id")
            user_id = (metadata or {}).get("user_id")

            kb_list = []
            # if company_id and user_id:
            #     logger.info(f"📚 获取公司 {company_id} 的知识库列表...")
            #     kb_list = await get_knowledge_bases(
            #         company_id=company_id,
            #         user_id=user_id
            #     )
            #     logger.info(f"📚 获取到 {len(kb_list)} 个知识库")
            kb_list = await get_knowledge_bases()

            # 创建上下文
            context = AgentContext(
                grag_id=grag_id,
                question=question,
                conversation_history=conversation_history or [],
                metadata={
                    **(metadata or {}),
                    "kb_list": kb_list,
                    "all_kbs": kb_list  # 传递完整知识库列表
                }
            )

            # 选择 Agent
            if agent_name:
                # 使用指定的 Agent
                agent = agent_registry.get_agent(agent_name)
                if not agent:
                    logger.error(f"❌ Agent '{agent_name}' 不存在")
                    return self._error_response(
                        f"Agent '{agent_name}' 不存在",
                        time.time() - start_time
                    )
                logger.info(f"✅ 使用指定的 Agent: {agent_name}")
            else:
                # 自动规划
                logger.info("📋 开始任务规划...")
                planned_agent_name = await agent_planner.plan(context)
                if not planned_agent_name:
                    # 如果规划失败，尝试自动查找
                    logger.info("⚙️ 规划失败，尝试自动查找合适的 Agent...")
                    agent = agent_registry.find_suitable_agent(context)
                else:
                    agent = agent_registry.get_agent(planned_agent_name)

                if not agent:
                    logger.error("❌ 未找到合适的 Agent")
                    return self._error_response(
                        "未找到合适的 Agent 处理此请求",
                        time.time() - start_time
                    )

            # 执行 Agent
            logger.info(f"🚀 执行 Agent: {agent.name}")
            result = await agent.execute(context)

            execution_time = time.time() - start_time

            # 格式化响应
            response = self._format_response(result, agent.name, execution_time)

            logger.info(f"✅ Agent 执行完成，总耗时: {execution_time:.2f}秒")
            logger.info("=" * 60)

            return response

        except Exception as e:
            logger.error(f"❌ Agent服务处理失败: {e}", exc_info=True)
            logger.info("=" * 60)
            return self._error_response(
                f"Agent服务处理失败: {str(e)}",
                time.time() - start_time
            )

    def _format_response(
            self,
            result: AgentResult,
            agent_name: str,
            total_time: float
    ) -> Dict[str, Any]:
        """格式化响应"""
        return {
            "success": result.success,
            "message": result.message,
            "data": result.data,
            "error": result.error,
            "agent_used": agent_name,
            "execution_time": total_time,
            "metadata": result.metadata
        }

    def _error_response(self, error: str, execution_time: float) -> Dict[str, Any]:
        """生成错误响应"""
        return {
            "success": False,
            "message": "处理失败",
            "data": None,
            "error": error,
            "agent_used": None,
            "execution_time": execution_time,
            "metadata": {}
        }

    def list_available_agents(self) -> Dict[str, Any]:
        """列出所有可用的 Agent"""
        agents = agent_registry.list_agents()
        capabilities = agent_registry.get_all_capabilities()

        return {
            "agents": agents,
            "capabilities": capabilities,
            "total": len(agents)
        }


# 全局 Agent 服务实例
agent_service = AgentService()
