"""
所有查询接口
"""
from fastapi import APIRouter, HTTPException, status
from models.schemas import R, ToGQueryRequest, GraphRAGQueryRequest, ToGGraphRAGQueryRequest, SiliconFlowQueryRequest
from services.query_tog_service import ToGService
from services.query_graphrag_service import GraphRAGService
from services.query_hybrid_service import HybridQueryService
from core.llm_client import llm_client
from utils.logger import logger, log_step
import time

router = APIRouter(prefix="/query", tags=["query"])


@router.get("/CORS_test", response_model=R)
async def test_cors():
    """简单的测试接口"""
    logger.info("收到 CORS跨域 测试请求")
    return R.ok(message="CORS test successful")


@router.post("/tog", response_model=R)
async def query_with_tog(request: ToGQueryRequest):
    """使用ToG (Think-on-Graph) 方法查询知识图谱"""
    try:
        logger.info("=" * 60)
        logger.info(f"[{request.grag_id}] 🔍 收到ToG查询请求")

        # 解析问题
        question = None
        if request.message_items:
            for message in reversed(request.message_items):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return R.error(message=error_msg, error_detail="没有user消息", code="400")

        logger.info(f"[{request.grag_id}] 💬 问题: {question}")

        # 执行ToG推理
        log_step(1, 3, "初始化ToG推理引擎", request.grag_id)
        tog_service = ToGService(
            grag_id=request.grag_id,
            max_depth=request.max_depth or 1,
            max_width=request.max_width or 5
        )

        log_step(2, 3, "执行ToG推理", request.grag_id)
        result = tog_service.reason(question)

        log_step(3, 3, "返回结果", request.grag_id)
        logger.info(f"[{request.grag_id}] ✅ 查询完成，耗时: {result['execution_time']:.2f}秒")
        logger.info("=" * 60)

        return R.ok(
            message="查询成功",
            data={
                "question": question,
                "answer": result["answer"],
                "execution_time": result["execution_time"],
                "grag_id": request.grag_id
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request.grag_id}] ❌ 查询处理失败: {e}", exc_info=True)
        return R.error(message="查询处理失败", error_detail=str(e), code="500")


@router.post("/graphrag", response_model=R)
async def query_graphrag(request: GraphRAGQueryRequest):
    """执行GraphRAG查询"""
    try:
        logger.info("=" * 60)
        logger.info(f"[{request.grag_id}] 🔍 收到GraphRAG查询请求")

        # 解析问题
        question = None
        if request.message_items:
            for message in reversed(request.message_items):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return R.error(message=error_msg, error_detail="没有user消息", code="400")

        logger.info(f"[{request.grag_id}] 💬 问题: {question}")

        # 执行GraphRAG查询
        log_step(1, 2, "执行GraphRAG查询", request.grag_id)
        service = GraphRAGService(grag_id=request.grag_id)
        success, answer, execution_time = service.query(question, request.method or "local")

        if success:
            logger.info(f"[{request.grag_id}] ✅ 查询成功，耗时: {execution_time:.2f}秒")
            logger.info("=" * 60)
            return R.ok(
                message="查询成功",
                data={
                    "question": question,
                    "answer": answer,
                    "grag_id": request.grag_id,
                    "execution_time": execution_time
                }
            )
        else:
            logger.error(f"[{request.grag_id}] ❌ 查询失败: {answer}")
            logger.info("=" * 60)
            return R.fail(message="查询失败", data={"error": answer}, code="500")

    except Exception as e:
        logger.error(f"[{request.grag_id}] ❌ 查询异常: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(message="查询处理失败", error_detail=str(e), code="500")


@router.post("/tog_grag", response_model=R)
async def query_tog_graphrag(request: ToGGraphRAGQueryRequest):
    """使用ToG和GraphRAG混合方法查询知识图谱"""
    try:
        logger.info("=" * 60)
        logger.info(f"[{request.grag_id}] 🔍 收到ToG+GraphRAG混合查询请求")

        # 解析问题
        question = None
        if request.message_items:
            for message in reversed(request.message_items):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return R.error(message=error_msg, error_detail="没有user消息", code="400")

        logger.info(f"[{request.grag_id}] 💬 问题: {question}")

        # 执行混合查询
        service = HybridQueryService(
            grag_id=request.grag_id,
            max_depth=request.max_depth or 5,
            max_width=request.max_width or 5,
            method=request.method or "local"
        )

        result = await service.query(question)

        if result["success"]:
            logger.info(f"[{request.grag_id}] ✅ 混合查询完成，总耗时: {result['execution_time']:.2f}秒")
            logger.info("=" * 60)
            return R.ok(
                message="混合查询成功",
                data={
                    "question": question,
                    "final_answer": result["final_answer"],
                    "tog_answer": result["tog_answer"],
                    "graphrag_answer": result["graphrag_answer"],
                    "grag_id": request.grag_id,
                    "execution_time": result["execution_time"]
                }
            )
        else:
            logger.error(f"[{request.grag_id}] ❌ {result['error']}")
            logger.info("=" * 60)
            return R.fail(message=result["error"], code="500")

    except Exception as e:
        logger.error(f"[{request.grag_id}] ❌ 混合查询处理失败: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(message="查询处理失败", error_detail=str(e), code="500")


@router.post("/siliconflow", response_model=R)
async def query_with_siliconflow(request: SiliconFlowQueryRequest):
    """直接调用硅基流动API进行问答"""
    try:
        logger.info("=" * 60)
        logger.info("🔍 收到硅基流动API查询请求")
        logger.info(f"💬 问题: {request.question}")

        # 记录开始时间
        start_time = time.time()

        # 配置参数（后端写死）
        system_prompt = """你是一个专业的运维问答AI助手，专注于为用户提供运维相关的技术支持和解答。

回答要求：
1. 回答必须采用分点陈述的方式，使用数字编号（1. 2. 3. ...）来组织内容
2. 每个要点应该简洁明了，条理清晰
3. 对于运维技术问题，提供具体的操作步骤和解决方案
4. 如果问题涉及故障排查，按照"问题描述 → 可能原因 → 排查步骤 → 解决方案"的逻辑进行回答

身份说明：
- 当用户问"你是什么模型"、"你是谁"、"介绍自己"等问题时，请回答："我是运维问答AI助手，专注于为您提供运维技术支持。"
- 当用户询问你基于什么模型、使用什么技术时，可以适当透露你使用了DeepSeek等先进的大语言模型技术，但要强调这是为了更好地为运维场景服务

请始终保持专业、友好、务实的态度，为用户提供有价值的运维建议。"""
        temperature = 0.3
        max_tokens = 3000

        # 调用硅基流动API
        log_step(1, 2, "调用硅基流动API")
        answer = llm_client.chat_with_siliconflow(
            prompt=request.question,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )

        # 计算耗时
        execution_time = time.time() - start_time

        if answer:
            log_step(2, 2, "返回结果")
            logger.info(f"✅ 查询成功，耗时: {execution_time:.2f}秒")
            logger.info("=" * 60)

            return R.ok(
                message="查询成功",
                data={
                    "question": request.question,
                    "answer": answer,
                    "execution_time": execution_time,
                }
            )
        else:
            logger.error("❌ 硅基流动API返回空响应")
            logger.info("=" * 60)
            return R.fail(message="API返回空响应", code="500")

    except Exception as e:
        logger.error(f"❌ 硅基流动API查询失败: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(message="查询处理失败", error_detail=str(e), code="500")