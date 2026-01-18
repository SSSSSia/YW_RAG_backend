"""
Agent 查询接口
"""
from fastapi import APIRouter, HTTPException
from models.schemas import R, AgentChatRequest
from services.agent_service import agent_service
from utils.logger import logger

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=R)
async def agent_chat(request: AgentChatRequest):
    """
    Agent 统一对话接口

    功能：
    1. 通过 AgentService 统一调度和管理
    2. 支持指定 Agent 或自动规划选择
    3. 自动获取可用知识库列表
    4. 如果指定知识库查询失败，自动回退查询其他知识库
    5. 支持问题复杂度分析和方法自动选择

    参数说明：
    - user_id: 用户ID（必填）
    - company_id: 公司ID（可选）
    - grag_id: 知识库ID（可选，不传则查询所有可用知识库）
    - message_items: 对话消息列表
    """
    try:
        logger.info("=" * 60)
        logger.info(f"[AgentRoutes] 🤖 收到Agent对话请求")
        logger.info(f"[AgentRoutes] 👤 用户ID: {request.user_id}")
        logger.info(f"[AgentRoutes] 🏢 公司ID: {request.company_id}")
        logger.info(f"[AgentRoutes] 📚 指定知识库ID: {request.grag_id or '未指定（全库查询）'}")

        # 1. 解析用户问题
        question = None
        if request.message_items:
            for message in reversed(request.message_items):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"[AgentRoutes] ❌ {error_msg}")
            return R.error(
                message=error_msg,
                error_detail="message_items中没有user消息",
                code="400"
            )

        logger.info(f"[AgentRoutes] 💬 用户问题: {question}")

        # 2. 构建对话历史
        conversation_history = []
        if request.message_items:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.message_items
            ]

        # 3. 构建 metadata
        metadata = {
            "user_id": request.user_id,
            "company_id": request.company_id,
        }

        # 4. 【核心】调用 AgentService 处理请求
        #    注意：AgentService 内部会：
        #    - 自动调用 get_knowledge_bases 获取所有可用知识库
        #    - 将知识库列表传递给 Agent
        #    - Agent 内部会实现优先查询指定 grag_id，失败后回退查询其他知识库的逻辑
        service_result = await agent_service.process_request(
            grag_id=request.grag_id or "default",
            question=question,
            conversation_history=conversation_history,
            metadata=metadata,
            agent_name=None
        )

        # 5. 处理响应
        if service_result.get("success"):
            logger.info(f"[AgentRoutes] ✅ 查询成功")
            logger.info(f"[AgentRoutes] 🤖 使用的 Agent: {service_result.get('agent_used')}")
            logger.info(f"[AgentRoutes] ⏱️ 执行时间: {service_result.get('execution_time', 0):.2f}秒")

            # 构建 API 响应数据
            response_data = {
                "question": question,
                "answer": service_result.get("data", {}).get("answer"),
                "method_used": service_result.get("metadata", {}).get("method"),
                "complexity": service_result.get("metadata", {}).get("complexity"),
                "execution_time": service_result.get("execution_time", 0),
                "user_id": request.user_id,
                "agent_used": service_result.get("agent_used")
            }

            # 添加知识库相关信息
            if "grag_id" in service_result.get("data", {}):
                response_data["grag_id"] = service_result["data"]["grag_id"]

            if "kb_name" in service_result.get("metadata", {}):
                response_data["kb_name"] = service_result["metadata"]["kb_name"]

            # 判断是否使用了回退查询
            if request.grag_id and "grag_id" in service_result.get("data", {}):
                response_data["fallback_used"] = (
                        request.grag_id != service_result["data"]["grag_id"]
                )

            logger.info("=" * 60)
            return R.ok(
                message=service_result.get("message", "查询成功"),
                data=response_data
            )
        else:
            logger.error(f"[AgentRoutes] ❌ 查询失败: {service_result.get('error')}")
            logger.info("=" * 60)
            return R.fail(
                message=service_result.get("message", "查询失败"),
                data={
                    "error": service_result.get("error"),
                    "question": question,
                    "agent_used": service_result.get("agent_used")
                },
                code="500"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AgentRoutes] ❌ 处理异常: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(
            message="Agent处理失败",
            error_detail=str(e),
            code="500"
        )


@router.post("/chat/{agent_name}", response_model=R)
async def agent_chat_with_name(agent_name: str, request: AgentChatRequest):
    """
    使用指定 Agent 的对话接口

    功能与 /chat 相同，但强制使用指定的 Agent
    """
    try:
        logger.info("=" * 60)
        logger.info(f"[AgentRoutes] 🤖 收到指定Agent对话请求")
        logger.info(f"[AgentRoutes] 🎯 指定Agent: {agent_name}")

        # 解析用户问题
        question = None
        if request.message_items:
            for message in reversed(request.message_items):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            return R.error(message="未找到有效的用户问题", code="400")

        logger.info(f"[AgentRoutes] 💬 用户问题: {question}")

        # 构建对话历史和 metadata
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in (request.message_items or [])
        ]

        metadata = {
            "user_id": request.user_id,
            "company_id": request.company_id,
        }

        # 调用 AgentService，指定 agent_name
        service_result = await agent_service.process_request(
            grag_id=request.grag_id or "default",
            question=question,
            conversation_history=conversation_history,
            metadata=metadata,
            agent_name=agent_name  # 【关键】指定使用哪个 Agent
        )

        # 处理响应
        if service_result.get("success"):
            logger.info(f"[AgentRoutes] ✅ 查询成功")

            response_data = {
                "question": question,
                "answer": service_result.get("data", {}).get("answer"),
                "method_used": service_result.get("metadata", {}).get("method"),
                "complexity": service_result.get("metadata", {}).get("complexity"),
                "execution_time": service_result.get("execution_time", 0),
                "user_id": request.user_id,
                "agent_used": agent_name
            }

            if "grag_id" in service_result.get("data", {}):
                response_data["grag_id"] = service_result["data"]["grag_id"]

            if "kb_name" in service_result.get("metadata", {}):
                response_data["kb_name"] = service_result["metadata"]["kb_name"]

            logger.info("=" * 60)
            return R.ok(
                message=service_result.get("message", "查询成功"),
                data=response_data
            )
        else:
            logger.error(f"[AgentRoutes] ❌ 查询失败")
            logger.info("=" * 60)
            return R.fail(
                message=service_result.get("message", "查询失败"),
                data={
                    "error": service_result.get("error"),
                    "question": question,
                    "agent_used": agent_name
                },
                code="500"
            )

    except Exception as e:
        logger.error(f"[AgentRoutes] ❌ 处理异常: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(
            message="Agent处理失败",
            error_detail=str(e),
            code="500"
        )


@router.get("/health", response_model=R)
async def agent_health():
    """
    Agent服务健康检查
    """
    try:
        # 通过 AgentService 检查可用 Agent
        agents_info = agent_service.list_available_agents()

        if agents_info.get("total", 0) > 0:
            return R.ok(
                message="Agent服务运行正常",
                data={
                    "status": "healthy",
                    "total_agents": agents_info.get("total"),
                    "available_agents": agents_info.get("agents"),
                    "capabilities": agents_info.get("capabilities")
                }
            )
        else:
            return R.fail(
                message="Agent服务没有可用的Agent",
                code="503"
            )
    except Exception as e:
        return R.error(
            message="Agent服务健康检查失败",
            error_detail=str(e),
            code="503"
        )


@router.get("/agents", response_model=R)
async def list_agents():
    """
    列出所有可用的 Agent 及其能力
    """
    try:
        agents_info = agent_service.list_available_agents()

        return R.ok(
            message="获取Agent列表成功",
            data={
                "agents": agents_info.get("agents"),
                "total": agents_info.get("total"),
                "capabilities": agents_info.get("capabilities")
            }
        )
    except Exception as e:
        logger.error(f"获取Agent列表失败: {e}", exc_info=True)
        return R.error(
            message="获取Agent列表失败",
            error_detail=str(e),
            code="500"
        )
