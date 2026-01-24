"""
AI审计和AI总结接口 - 修改版
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import R, SummaryRequest, AlarmData, WorkOrderData
from core.mysql_db import get_operation_db
from services.session_storage_service import get_session_storage_service
from core.llm_client import llm_client
from utils.logger import logger, log_step
from datetime import datetime
from typing import Optional
import json
import re

router = APIRouter(prefix="/yw", tags=["运维AI审计和总结"])


@router.post("/check", response_model=R)
async def ai_check(
    pic: UploadFile = File(..., description="图片文件"),
    sessionID: str = Form(..., description="会话ID（设备ID）"),
    operation: str = Form(..., description="图片对应的操作")
):
    """
    AI审计接口 - 根据图片和操作判断是否存在危险并生成告警信息

    请求参数（multipart/form-data）：
    - sessionID: 会话ID（设备ID）
    - pic: 图片文件（YYYYMMDDHHmmss命名）
    - operation: 图片对应的操作描述

    返回：
    - 如果无告警：code="200", message="操作正常", data包含设备编号和工作内容
    - 如果有告警：code="500", message="发现安全风险", data包含告警信息
      - equipment_asset: 设备编号（即sessionID）
      - alarm: 告警信息（由LLM判断危险性生成）
      - alarm_time: 告警时间
      - work_content: 工作内容摘要
    """
    try:
        logger.info("=" * 60)
        logger.info(f"[{sessionID}] 🔍 收到AI审计请求")
        logger.info(f"[{sessionID}] 操作: {operation}")
        logger.info(f"[{sessionID}] 图片文件: {pic.filename}")

        # 读取图片数据
        image_data = await pic.read()

        # 保存图片到会话目录
        log_step(1, 4, "保存图片", sessionID)
        filename = pic.filename or f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        image_path = get_session_storage_service().save_image(sessionID, filename, image_data)

        # 读取图片并转换为base64
        log_step(2, 4, "读取图片并准备LLM分析", sessionID)
        image_base64 = get_session_storage_service().get_image_base64(image_path)
        if not image_base64:
            logger.error(f"[{sessionID}] ❌ 图片读取失败")
            return R.error(message="图片读取失败", code="500")

        # 构建AI审计的提示词 - 结构化输出
        system_prompt = """你是一个专业的运维安全审计AI助手。你的任务是判断运维操作是否存在安全风险。

【判断标准】
**高危操作（has_risk=true, risk_level="high"）**：
- 删除关键数据、格式化磁盘、停止核心服务
- 修改生产环境核心配置（防火墙、数据库、网络、系统配置）
- 执行未知来源的脚本或命令
- 危险命令：rm -rf /、dd、shutdown、format等
- 未在权限范围内的敏感操作

**中危操作（has_risk=true, risk_level="medium"）**：
- 修改非关键配置
- 重启非核心服务
- 可能影响性能的操作
- 操作不规范但未造成明显风险

**低危操作（has_risk=true, risk_level="low"）**：
- 轻微操作不规范
- 潜在风险很小

**安全操作（has_risk=false, risk_level="none"）**：
- 查询类操作（ls、cat、grep、select等）
- 常规维护操作
- 正常的配置查看 

【输出要求】
必须严格返回JSON格式（不要使用markdown代码块）：
{
  "has_risk": true或false,
  "risk_level": "high/medium/low/none",
  "alarm_message": "具体告警内容（仅has_risk为true时填写，20-100字）"
}

注意：
- has_risk为false时，risk_level必须为"none"，alarm_message留空或填"无风险"
- has_risk为true时，必须明确说明具体风险点"""

        user_prompt = f"""请审计以下运维操作：

操作描述：{operation}

请结合截图内容判断风险等级，并严格按JSON格式返回结果。"""

        # 调用视觉LLM进行安全审计
        log_step(3, 4, "调用视觉LLM进行安全审计", sessionID)
        alarm_message = llm_client.chat_with_vision(
            prompt=user_prompt,
            image_base64=image_base64,
            temperature=0.1,
            max_tokens=500,
            system_prompt=system_prompt
        )

        if not alarm_message:
            logger.warning(f"[{sessionID}] ⚠️ LLM调用失败")
            return R.error(message="AI分析失败", code="500")

        # 解析LLM返回的JSON结果
        log_step(4, 6, "解析AI审计结果", sessionID)
        try:
            # 清理可能的markdown代码块标记
            response_text = alarm_message.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            audit_result = json.loads(response_text)
            has_risk = audit_result.get("has_risk", False)
            risk_level = audit_result.get("risk_level", "none")
            alarm_content = audit_result.get("alarm_message", "")

            logger.info(f"[{sessionID}] 📊 AI审计结果: has_risk={has_risk}, risk_level={risk_level}")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{sessionID}] ⚠️ JSON解析失败: {e}, 原始响应: {alarm_message}")
            # 解析失败时，保守策略：如果有明显的问题关键词则告警
            error_keywords = ["删除", "格式化", "shutdown", "rm -rf", "drop", "truncate"]
            has_risk = any(keyword in alarm_message.lower() for keyword in error_keywords)
            risk_level = "medium" if has_risk else "none"
            alarm_content = alarm_message if has_risk else ""

        # 生成操作的简要总结（用于保存到数据库，供后续summary使用）
        log_step(5, 6, "生成操作总结", sessionID)
        summary_prompt = f"请用一句话（30字内）概括这个操作：{operation}"
        summary = llm_client.chat_with_vision(
            prompt=summary_prompt,
            image_base64=image_base64,
            temperature=0.1,
            max_tokens=200,
            system_prompt="你是一个运维操作记录助手，请简洁概括操作内容。"
        ) or operation

        # 保存到MySQL数据库
        log_step(6, 6, "保存操作记录到MySQL数据库", sessionID)
        get_operation_db().save_record(
            session_id=sessionID,
            operation=operation,
            image_path=image_path,
            summary=summary
        )

        # 根据审计结果返回响应
        if has_risk and risk_level != "none":
            # 有告警：返回code=300001
            alarm_time = datetime.now()
            result_data = AlarmData(
                equipment_asset=sessionID,
                alarm=alarm_content or "检测到安全风险",
                alarm_time=alarm_time,
                work_content=summary,
                risk_level=risk_level
            )

            logger.info(f"[{sessionID}] ⚠️ 发现安全风险 [{risk_level.upper()}]")
            logger.info(f"[{sessionID}] 告警信息: {alarm_content}")
            logger.info(f"[{sessionID}] 工作内容: {summary}")
            logger.info("=" * 60)

            return R.error(
                message="发现安全风险",
                code="300001",
                data=result_data.model_dump()
            )
        else:
            # 无告警：返回code=200
            logger.info(f"[{sessionID}] ✅ 操作正常，无安全风险")
            logger.info(f"[{sessionID}] 工作内容: {summary}")
            logger.info("=" * 60)

            return R.ok(
                message="操作正常",
                data={"equipment_asset": sessionID, "work_content": summary}
            )

    except Exception as e:
        logger.error(f"[{sessionID}] ❌ AI审计处理失败: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(message="审计处理失败", data=str(e), code="500")


@router.post("/summary", response_model=R)
async def ai_summary(request: SummaryRequest):
    """
    AI总结接口 - 根据会话中的所有操作记录（包括图片）生成详细工单信息

    请求参数：
    - sessionID: 会话ID（设备ID）

    返回：
    - ds_id: 设备ID（int类型，从sessionID转换而来）
    - work_class: 工单分类（1=软件，2=硬件）
    - work_notice: 工作内容详细总结
    """
    try:
        logger.info("=" * 60)
        logger.info(f"[{request.sessionID}] 🔍 收到AI总结请求")

        # 从数据库获取会话的所有操作记录
        log_step(1, 4, "从数据库获取会话操作记录", request.sessionID)
        records = get_operation_db().get_records_by_session(request.sessionID)

        if not records:
            logger.warning(f"[{request.sessionID}] ⚠️ 未找到操作记录")
            return R.fail(message="未找到操作记录，无法生成工单", code="400")

        logger.info(f"[{request.sessionID}] 找到 {len(records)} 条操作记录")

        # 读取所有图片并转换为base64（用于传给大模型）
        log_step(2, 4, "读取操作记录图片", request.sessionID)
        image_data_list = []
        for record in records:
            image_path = record.get('image_path')
            if image_path:
                image_base64 = get_session_storage_service().get_image_base64(image_path)
                if image_base64:
                    image_data_list.append({
                        'operation': record['operation'],
                        'summary': record['summary'],
                        'image': image_base64,
                        'time': record['created_at']
                    })

        # 构建操作摘要文本
        operations_summary = []
        for idx, record in enumerate(records, 1):
            operations_summary.append(
                f"操作{idx}:\n"
                f"- 操作描述: {record['operation']}\n"
                f"- 总结: {record['summary']}\n"
                f"- 时间: {record['created_at']}"
            )

        operations_text = "\n\n".join(operations_summary)

        # 构建AI总结的提示词 - 强调工作内容要详细
        system_prompt = """你是一个专业的运维工单生成AI助手。你的任务是根据运维操作记录（包括图片和文字描述）生成详细的工单信息。

工单分类说明（work_class）：
- 1: 软件（涉及软件安装、配置、调试、升级等）
- 2: 硬件（涉及硬件设备维护、更换、维修等）

**工作内容（work_notice）要求：**
- 必须详细描述所有操作步骤
- 包含具体的设备、软件、配置信息
- 说明操作目的和结果
- 字数要求：至少150字，确保信息完整

响应格式要求（必须是JSON格式）：
{
    "work_class": 工单分类（整数，1=软件，2=硬件）,
    "work_notice": "详细的工作内容总结（至少150字，包含所有操作细节）"
}

请始终返回有效的JSON格式。"""

        user_prompt = f"""请根据以下运维操作记录生成详细的工单信息：

会话ID（设备ID）: {request.sessionID}
共有 {len(records)} 条操作记录。

操作记录详情：
{operations_text}

注意：除了上述文字信息，我还会提供相关的操作截图图片。

请综合分析：
1. 判断主要是软件操作还是硬件操作
2. **生成详细的工作内容总结（至少150字）**，包括：
   - 所有操作步骤
   - 涉及的设备和组件
   - 配置修改内容
   - 操作目的和结果

请按照要求的JSON格式返回结果。注意：ds_id将由系统从sessionID中提取。"""

        # 调用LLM进行总结（支持多图片）
        log_step(3, 4, "调用LLM生成详细工单信息", request.sessionID)

        # 如果有图片，使用视觉模型；否则使用文本模型
        if image_data_list:
            # 使用第一张图片作为代表（或者可以修改为支持多图）
            llm_response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_data_list[0]['image'],
                temperature=0.3,
                max_tokens=2000,  # 增加token数以支持详细描述
                system_prompt=system_prompt
            )
        else:
            llm_response = llm_client.chat_with_siliconflow(
                prompt=user_prompt,
                temperature=0.3,
                max_tokens=2000,
                system_prompt=system_prompt
            )

        if not llm_response:
            logger.error(f"[{request.sessionID}] ❌ LLM调用失败")
            return R.error(message="AI分析失败", code="500")

        # 解析LLM响应
        try:
            # 尝试提取JSON内容
            response_text = llm_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            # ds_id直接从sessionID转换（提取数字部分）
            numbers = re.findall(r'\d+', request.sessionID)
            ds_id = int(numbers[0]) if numbers else int(request.sessionID)

            work_order = WorkOrderData(
                ds_id=ds_id,
                work_class=int(result.get("work_class", 1)),  # 默认为软件
                work_notice=result.get("work_notice", "运维操作总结")
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[{request.sessionID}] ⚠️ JSON解析失败，使用默认值: {e}")
            # 如果解析失败，使用默认值
            numbers = re.findall(r'\d+', request.sessionID)
            ds_id = int(numbers[0]) if numbers else int(request.sessionID)

            # 至少提供基本的操作汇总
            basic_summary = "运维操作包括：" + "；".join([r['summary'] for r in records[:5]])

            work_order = WorkOrderData(
                ds_id=ds_id,
                work_class=1,
                work_notice=basic_summary
            )

        log_step(4, 4, "工单信息生成完成", request.sessionID)
        logger.info(f"[{request.sessionID}] ✅ AI总结完成")
        logger.info(f"[{request.sessionID}] 工单信息: ds_id={work_order.ds_id}, work_class={work_order.work_class}（{'软件' if work_order.work_class == 1 else '硬件'}）")
        logger.info(f"[{request.sessionID}] 工作内容长度: {len(work_order.work_notice)}字")
        logger.info("=" * 60)

        return R.ok(
            message="总结完成",
            data=work_order.model_dump()
        )

    except Exception as e:
        logger.error(f"[{request.sessionID}] ❌ AI总结处理失败: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(message="总结处理失败", data=str(e), code="500")