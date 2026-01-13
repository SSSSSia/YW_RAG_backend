"""
所有查询接口
"""
from fastapi import APIRouter, HTTPException, status
from models.schemas import R, ToGQueryRequest, GraphRAGQueryRequest, ToGGraphRAGQueryRequest
from services.query_tog_service import ToGService
from services.query_graphrag_service import GraphRAGService
from services.query_hybrid_service import HybridQueryService
from utils.logger import logger, log_step

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
        if request.messages:
            for message in reversed(request.messages):
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
            max_depth=request.max_depth or 10,
            max_width=request.max_width or 3
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
        if request.messages:
            for message in reversed(request.messages):
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
        if request.messages:
            for message in reversed(request.messages):
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
            max_depth=request.max_depth or 10,
            max_width=request.max_width or 3,
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