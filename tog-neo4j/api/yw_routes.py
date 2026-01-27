"""
AI审计和AI总结接口
"""
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from models.schemas import R, SummaryRequest
from services.yw_service import get_audit_service
from utils.logger import logger

router = APIRouter(prefix="/yw", tags=["运维AI审计和总结"])


@router.post("/check", response_model=R)
async def ai_check(
    pic: UploadFile = File(..., description="图片文件"),
    sessionID: str = Form(..., description="会话ID（设备ID）"),
    operation: str = Form(..., description="图片对应的操作（JSON字符串）"),
    process_name: Optional[str] = Form(None, description="预设流程名称（可选）")
):
    """
    AI审计接口 - 基于操作流程的智能审计

    请求参数（multipart/form-data）：
    - sessionID: 会话ID（设备ID）
    - pic: 图片文件（YYYYMMDDHHmmss命名）
    - operation: 图片对应的操作描述（JSON字符串，AuditOpt对象）
    - process_name: 预设流程名称（可选，用于演示/测试，跳过LLM识别）

    返回：
    - code="200": 操作正常（在流程内且无风险）
    - code="200001": 轻微告警（跳出流程但无风险）
    - code="300001": 严重告警（跳出流程且有风险）

    说明：
    - 如果提供 process_name，将直接使用该流程，跳过LLM识别
    - 如果不提供 process_name，则调用LLM自动识别流程
    """
    try:
        logger.info(f"[YWRoutes] 收到AI审计请求，sessionID: {sessionID}, 图片: {pic.filename}")

        # 读取图片数据
        image_data = await pic.read()

        # 调用 AuditService 处理
        result = await get_audit_service().ai_check(
            pic_filename=pic.filename,
            image_data=image_data,
            sessionID=sessionID,
            operation=operation,
            process_name=process_name or "系统重装"
        )

        logger.info(f"[YWRoutes] AI审计完成，sessionID: {sessionID}")
        return result

    except Exception as e:
        logger.error(f"[YWRoutes] AI审计处理失败: {e}", exc_info=True)
        return R.error(message="审计处理失败", data=str(e), code="500")


@router.post("/summary", response_model=R)
async def ai_summary(request: SummaryRequest):
    """
    AI总结接口 - 根据会话中的所有操作记录生成详细工单信息

    请求参数：
    - sessionID: 会话ID（设备ID）

    返回：
    - ds_id: 设备ID（str类型，直接使用sessionID）
    - work_class: 工单分类（1=软件，2=硬件）
    - work_notice: 工作内容详细总结

    说明：
    - 使用Neo4j中的操作流程信息和MySQL中的操作summary生成工单
    - 不传递图片，避免上下文过长，提高处理速度和准确性
    """
    try:
        logger.info(f"[YWRoutes] 收到AI总结请求，sessionID: {request.sessionID}")

        # 调用 AuditService 处理
        result = await get_audit_service().ai_summary(request)

        logger.info(f"[YWRoutes] AI总结完成，sessionID: {request.sessionID}")
        return result

    except Exception as e:
        logger.error(f"[YWRoutes] AI总结处理失败: {e}", exc_info=True)
        return R.error(message="总结处理失败", data=str(e), code="500")
