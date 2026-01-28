"""
AI审计和AI总结接口 - 修改版
"""
import json
import re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import R, SummaryRequest, AlarmData, WorkOrderData
from services import get_audit_service, get_session_storage_service
from core import llm_client
from core.mysql_db import get_operation_db
from utils.logger import logger, log_step
from typing import Optional

router = APIRouter(prefix="/yw", tags=["运维AI审计和总结"])


def _parse_json_from_response(response: str) -> Optional[dict]:
    """
    从LLM响应中智能提取JSON对象

    容错处理：
    1. 自动移除markdown代码块标记（```json 和 ```）
    2. 处理前后空白字符
    3. 智能提取JSON对象（通过定位{和}）

    Args:
        response: LLM返回的原始响应

    Returns:
        解析后的JSON字典，失败返回None
    """
    try:
        if not response:
            return None

        response_text = response.strip()

        # 方法1: 智能提取 - 直接定位JSON对象的开始和结束
        # 找到第一个 { 或 [ 的位置
        json_start = -1
        for i, char in enumerate(response_text):
            if char in ['{', '[']:
                json_start = i
                break

        if json_start == -1:
            logger.warning(f"未找到JSON开始标记，原始响应前200字符: {response[:200]}")
            # 尝试传统方法：移除markdown代码块
            return _parse_json_traditional(response_text)

        # 找到最后一个 } 或 ] 的位置
        json_end = -1
        for i in range(len(response_text) - 1, -1, -1):
            if response_text[i] in ['}', ']']:
                json_end = i
                break

        if json_end == -1 or json_end <= json_start:
            logger.warning(f"未找到JSON结束标记，原始响应前200字符: {response[:200]}")
            # 尝试传统方法：移除markdown代码块
            return _parse_json_traditional(response_text)

        # 提取JSON内容
        json_content = response_text[json_start:json_end + 1]

        # 尝试解析
        result = json.loads(json_content)

        logger.debug(f"JSON解析成功，提取的长度: {len(json_content)}")
        return result

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        # 如果智能提取失败，尝试传统方法
        logger.debug(f"智能JSON提取失败，尝试传统方法: {e}")
        return _parse_json_traditional(response_text)


def _parse_json_traditional(response_text: str) -> Optional[dict]:
    """
    传统方法：移除markdown代码块后解析JSON

    Args:
        response_text: 已经strip过的响应文本

    Returns:
        解析后的JSON字典，失败返回None
    """
    try:
        # 尝试移除 ```json ... ``` 格式
        if "```json" in response_text:
            parts = response_text.split("```json")
            if len(parts) > 1:
                json_text = parts[1].split("```")[0].strip()
                if json_text:
                    result = json.loads(json_text)
                    logger.debug("使用```json标记解析成功")
                    return result

        # 尝试移除 ``` ... ``` 格式
        if "```" in response_text:
            parts = response_text.split("```")
            # 取第二个```块（如果有的话）
            if len(parts) >= 3:
                json_text = parts[1].strip()
                if json_text:
                    result = json.loads(json_text)
                    logger.debug("使用```标记解析成功")
                    return result
            elif len(parts) == 2:
                # 只有两个```，取中间的内容
                json_text = parts[1].strip()
                # 移除可能的lang标识（第一行）
                lines = json_text.split('\n', 1)
                if len(lines) > 1:
                    json_text = lines[1].strip()
                if json_text:
                    result = json.loads(json_text)
                    logger.debug("使用单个```标记解析成功")
                    return result

        # 最后尝试：直接解析整个响应
        if response_text:
            result = json.loads(response_text)
            logger.debug("直接解析成功")
            return result

        logger.warning(f"传统JSON解析也失败，响应前200字符: {response_text[:200]}")
        return None

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"传统JSON解析失败: {e}, 响应前200字符: {response_text[:200]}")
        return None


@router.post("/check", response_model=R)
async def ai_check(
    pic: UploadFile = File(..., description="图片文件"),
    sessionID: str = Form(..., description="会话ID（设备ID）"),
    operation: str = Form(..., description="图片对应的操作（JSON字符串）"),
    process_name: Optional[str] = Form(None, description="预设流程名称（可选）")
):
    """
    AI审计接口 - 基于操作流程的智能审计（无数据库版本）

    请求参数（multipart/form-data）：
    - sessionID: 会话ID（设备ID）
    - pic: 图片文件（YYYYMMDDHHmmss命名）
    - operation: 图片对应的操作描述（JSON字符串，AuditOpt对象）
    - process_name: 预设流程名称（可选，用于演示/测试）

    返回：
    - code="200": 操作正常（在流程内且无风险）
    - code="20001": 轻微告警（跳出流程但无风险）
    - code="30001": 严重告警（跳出流程且有风险）

    说明：
    - 如果提供 process_name，将使用该流程进行检查
    - 如果不提供 process_name，则只进行标准风险审计（不使用流程）
    - 此版本不使用MySQL数据库，只使用Neo4j（可选）和LLM进行审计
    """
    try:
        logger.info(f"[YWRoutes] 收到AI审计请求，sessionID: {sessionID}, 图片: {pic.filename}")

        # 读取图片数据
        image_data = await pic.read()

        # 调用 AuditService 处理
        # 注意：如果不提供 process_name，则只进行标准风险审计（不使用流程）
        result = await get_audit_service().ai_check(
            pic_filename=pic.filename,
            image_data=image_data,
            sessionID=sessionID,
            operation=operation,
            process_name=process_name
        )

        logger.info(f"[YWRoutes] AI审计完成，sessionID: {sessionID}")
        return result

    except Exception as e:
        logger.error(f"[YWRoutes] AI审计处理失败: {e}", exc_info=True)
        return R.error(message="审计处理失败", error_detail=str(e), code="500")


@router.post("/summary", response_model=R)
async def ai_summary(request: SummaryRequest):
    """
    AI总结接口 - 根据会话中的所有操作记录（包括图片）生成详细工单信息

    请求参数：
    - sessionID: 会话ID（设备ID）
    - delete: 是否在总结后删除该会话的所有记录（可选，默认false）

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

        # 构建操作摘要文本（第一轮：不加载图片）
        log_step(2, 5, "构建操作摘要", request.sessionID)
        operations_summary = []
        for idx, record in enumerate(records, 1):
            operations_summary.append(
                f"操作{idx}:\n"
                f"- 操作描述: {record['operation']}\n"
                f"- 总结: {record['summary']}\n"
            )

        operations_text = "\n\n".join(operations_summary)

        # ========== 第一轮：让LLM判断需要查看哪些关键操作的图片 ==========
        log_step(3, 5, "LLM智能选择需要查看的图片", request.sessionID)

        # 检查是否有图片可用
        has_images = any(record.get('image_path') for record in records)

        selected_image_indices = []

        if has_images:
            # 第一轮：只传文字，让LLM选择需要查看图片的操作序号
            selection_prompt = f"""请分析以下运维操作记录，判断需要查看哪些操作的截图才能准确生成工单。

会话ID（设备ID）: {request.sessionID}
共有 {len(records)} 条操作记录。

操作记录详情：
{operations_text}

**选择规则（最多选择5个操作）：**
1. 优先选择关键操作（如配置修改、软件安装、重要决策点）
2. 选择代表性操作（如开始、结束、重要转折点）
3. 选择复杂操作（文字描述不够清晰的操作）
4. 避免选择重复性操作

**输出格式（必须是JSON，不要使用markdown代码块）：**
{{
    "selected_operations": [1, 3, 5],  // 需要查看图片的操作序号列表（1-{len(records)}）
    "reason": "选择理由（30-50字）"
}}

注意：
- selected_operations是一个数字数组，表示操作序号
- 最多选择5个操作
- 如果操作记录简单明确，可以选择空数组[]"""

            try:
                selection_response = llm_client.chat_with_siliconflow(
                    prompt=selection_prompt,
                    temperature=0.1,
                    max_tokens=300,
                    system_prompt="你是一个智能图片选择助手，根据操作文字描述判断需要查看哪些操作的截图。"
                )

                if selection_response:
                    # 解析选择结果
                    selection_text = selection_response.strip()
                    if "```json" in selection_text:
                        selection_text = selection_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in selection_text:
                        selection_text = selection_text.split("```")[1].split("```")[0].strip()

                    selection_result = json.loads(selection_text)
                    selected_image_indices = selection_result.get("selected_operations", [])
                    reason = selection_result.get("reason", "")

                    # 限制最多5张图片
                    selected_image_indices = selected_image_indices[:5]

                    logger.info(f"[{request.sessionID}] LLM选择了 {len(selected_image_indices)} 张图片: {selected_image_indices}")
                    logger.info(f"[{request.sessionID}] 选择理由: {reason}")
                else:
                    logger.warning(f"[{request.sessionID}] 图片选择LLM调用失败，使用默认策略")
                    # 默认策略：选择第一张、最后一张和中间一张
                    selected_image_indices = [1]
                    if len(records) > 2:
                        selected_image_indices.append(len(records))
                        selected_image_indices.append(len(records) // 2 + 1)

            except Exception as e:
                logger.warning(f"[{request.sessionID}] 图片选择失败: {e}，使用默认策略")
                # 默认策略
                selected_image_indices = [1]
                if len(records) > 2:
                    selected_image_indices.append(len(records))

        # ========== 第二轮：根据选择加载对应的图片 ==========
        log_step(4, 5, "加载选中的图片", request.sessionID)

        # 根据选择的操作序号加载图片
        selected_images = []
        for idx in selected_image_indices:
            # 转换为0-based索引
            record_idx = idx - 1
            if 0 <= record_idx < len(records):
                record = records[record_idx]
                image_path = record.get('image_path')
                if image_path:
                    image_base64 = get_session_storage_service().get_image_base64(image_path)
                    if image_base64:
                        selected_images.append({
                            'index': idx,
                            'operation': record['operation'],
                            'summary': record['summary'],
                            'image': image_base64
                        })

        logger.info(f"[{request.sessionID}] 成功加载 {len(selected_images)} 张图片")

        # 构建AI总结的提示词
        system_prompt = """你是一个专业的运维工单生成AI助手。你的任务是根据运维操作记录（包括图片和文字描述）生成详细的工单信息。不用说明对应了哪张截图，只输出自然语言描述就行。

【正常操作流程白名单】
⚠️ 严格限制：只有以下操作被视为正常操作，任何超出或不属于这些步骤的操作都必须标记为异常！

1. 系统重装流程（仅限这17步）：
   1) 点击 "Test this media & install Kylin linux Advanced Server V11"
   2) 按Enter键
   3) 点击"中文-简体中文"
   4) 点击"继续"
   5) 点击"安装目的地（D）"
   6) 点击"完成（D）"
   7) 点击"Root账户"
   8) 输入Root密码
   9) 点击"确认(C)"的输入框
   10) 再次输入Root密码
   11) 点击左上角完成
   12) 点击"开始安装"
   13) 点击"Kylin Linux Advanced Server（6.6.0-32.7.ky11.x86_64) V11（Swan25）"
   14) 点击"许可信息（L）"
   15) 点击"我同意许可协议（A）"
   16) 点击"完成（D）"
   17) 点击"结束配置（F）"

2. 密码重置流程（仅限这8步）：
   1) 点击"Kylin Linux Advanced Server（6.6.0-32.7.ky11.x86_64) V11（Swan25）"
   2) 按Enter键
   3) 输入"passwd"命令
   4) 输入密码
   5) 按Enter键
   6) 再次输入密码
   7) 按Enter键
   8) 输入"/usr/sbin/reboot -f"强制重启生效

【异常操作识别规则 - 关键】
🔴 以下操作必须被标记为异常/违规，即使出现在流程之后：
1. 访问敏感目录：/root、/etc、/boot、/sys、/proc、/home/其他用户
2. 打开/编辑敏感文件：/etc/passwd、/etc/shadow、*.conf、*.cfg、私钥、脚本文件
3. 危险命令：删除（rm、delete）、格式化、停止服务、修改权限
4. 未授权操作：创建用户、安装软件、修改网络配置
5. 探索性操作：浏览文件系统、查看日志（除非是明确故障排查）

⚠️ 判断原则：
- 如果操作超出了上述两个白名单流程的范围，必须标记为异常
- 即使在流程完成后出现的任何操作，也需要明确标注为"额外操作"
- 不要假设任何未列出的操作是"正常"的

【总结输出格式要求】
工单内容必须分为两部分：
1. 【正常流程操作】：只包含上述白名单中的操作
2. 【异常/额外操作】：任何超出白名单的操作，详细描述并说明风险

示例格式：
"本次运维包括以下内容：
【正常流程操作】执行了系统重装流程，包括语言选择、磁盘配置、Root密码设置等17个步骤。
【异常/额外操作】在流程完成后，操作人员额外访问了/etc目录并打开了配置文件xxx.conf，这可能存在安全风险。"

工单分类说明（work_class）：
- 1: 软件（涉及软件安装、配置、调试、升级等）
- 2: 硬件（涉及硬件设备维护、更换、维修等）

**工作内容（work_notice）要求：**
- 必须详细描述所有操作步骤
- 如果提供了截图，请结合截图内容进行分析
- 包含具体的设备、软件、配置信息
- 说明操作目的和结果
- **严格区分正常操作和异常操作**
- 字数要求：至少150字，确保信息完整

响应格式要求（必须是JSON格式）：
{
    "work_class": 工单分类（整数，1=软件，2=硬件）,
    "work_notice": "详细的工作内容总结（至少150字，包含所有操作细节，必须区分正常和异常操作）"
}

请始终返回有效的JSON格式。"""

        user_prompt = f"""请根据以下运维操作记录生成详细的工单信息：

会话ID（设备ID）: {request.sessionID}
共有 {len(records)} 条操作记录。

操作记录详情：
{operations_text}"""

        user_prompt += """

请综合分析：
1. 判断主要是软件操作还是硬件操作
2. **生成详细的工作内容总结（至少150字）**，包括：
   - 所有操作步骤
   - 涉及的设备和组件
   - 配置修改内容
   - 操作目的和结果

请按照要求的JSON格式返回结果。注意：ds_id将由系统从sessionID中提取。"""

        # ========== 第三轮：调用LLM生成最终工单 ==========
        log_step(5, 5, "调用LLM生成详细工单信息", request.sessionID)

        # 如果有选中的图片，使用视觉模型；否则使用文本模型
        if selected_images:
            if len(selected_images) == 1:
                # 单张图片
                logger.info(f"[{request.sessionID}] 使用视觉模型（1张图片）")
                llm_response = llm_client.chat_with_vision(
                    prompt=user_prompt,
                    image_base64=selected_images[0]['image'],
                    temperature=0.3,
                    max_tokens=2000,
                    system_prompt=system_prompt
                )
            else:
                # 多张图片 - 使用新的多图方法
                logger.info(f"[{request.sessionID}] 使用多图视觉模型（{len(selected_images)}张图片）")
                images_base64 = [img['image'] for img in selected_images]
                llm_response = llm_client.chat_with_multiple_visions(
                    prompt=user_prompt,
                    images_base64=images_base64,
                    temperature=0.3,
                    max_tokens=2000,
                    system_prompt=system_prompt
                )
        else:
            # 没有图片，使用文本模型
            logger.info(f"[{request.sessionID}] 使用文本模型（无图片）")
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
            # 尝试提取JSON内容（使用智能提取逻辑）
            # logger.info(f"[{request.sessionID}]  LLM响应：{llm_response}")

            # 使用智能JSON提取方法
            result = _parse_json_from_response(llm_response)

            if not result:
                logger.warning(f"[{request.sessionID}] ⚠️ JSON提取失败")
                raise json.JSONDecodeError("JSON提取失败", "", 0)

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
        logger.info(f"[{request.sessionID}] 工作内容: {work_order.work_notice}")

        # ========== 删除会话记录（如果请求要求删除）==========
        if request.delete:
            log_step(5, 5, f"删除会话记录（delete={request.delete}）", request.sessionID)
            try:
                deleted_count = get_operation_db().delete_records_by_session(request.sessionID)
                logger.info(f"[{request.sessionID}] ✅ 已删除 {deleted_count} 条操作记录")
            except Exception as e:
                logger.error(f"[{request.sessionID}] ❌ 删除操作记录失败: {e}", exc_info=True)
                # 删除失败不影响总结结果，只记录错误

        logger.info("=" * 60)

        return R.ok(
            message="总结完成",
            data=work_order.model_dump()
        )

    except Exception as e:
        logger.error(f"[{request.sessionID}] ❌ AI总结处理失败: {e}", exc_info=True)
        logger.info("=" * 60)
        return R.error(message="总结处理失败", error_detail=str(e), code="500")