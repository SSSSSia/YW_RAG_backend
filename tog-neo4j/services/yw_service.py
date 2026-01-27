"""
AI审计和AI总结服务 - 封装运维操作的智能审计和工单生成逻辑
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from models.schemas import R, SummaryRequest, AlarmData, WorkOrderData, AuditOpt
from core.mysql_db import get_operation_db
from services.session_storage_service import get_session_storage_service
from core.neo4j_db import get_yw_neo4j
from core.llm_client import llm_client
from utils.logger import logger, log_step


class AuditService:
    """AI审计和总结服务"""

    def __init__(self):
        """初始化服务"""
        logger.info("✅ AuditService 初始化完成")

    async def ai_check(
        self,
        pic_filename: str,
        image_data: bytes,
        sessionID: str,
        operation: str,
        process_name: Optional[str] = None
    ) -> R:
        """
        AI审计 - 基于操作流程的智能审计

        Args:
            pic_filename: 图片文件名
            image_data: 图片数据
            sessionID: 会话ID（设备ID）
            operation: 图片对应的操作描述（JSON字符串）
            process_name: 预设流程名称（可选）

        Returns:
            R: 审计结果
        """
        try:
            logger.info("=" * 60)
            logger.info(f"[AuditService] [{sessionID}] 🔍 开始AI审计")

            # 解析operation JSON字符串
            audit_opt = self._parse_operation(sessionID, operation)
            logger.info(f"[AuditService] [{sessionID}] 事件类型: {audit_opt.event_type}")
            logger.info(f"[AuditService] [{sessionID}] 事件内容: {audit_opt.event_content}")

            # ========== 事件过滤：只处理"按下"事件，忽略"松开"事件 ==========
            if self._should_ignore_event(sessionID, audit_opt):
                logger.info(f"[AuditService] [{sessionID}] ⏭️ 忽略松开事件（state=false）")
                logger.info("=" * 60)
                return R.ok(message="忽略松开事件", data={"ignored": True, "reason": "state=false"})

            # 保存图片到会话目录
            log_step(1, 7, "保存图片", sessionID)
            filename = pic_filename or f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = get_session_storage_service().save_image(sessionID, filename, image_data)

            # 读取图片并转换为base64
            log_step(2, 7, "读取图片并准备LLM分析", sessionID)
            image_base64 = get_session_storage_service().get_image_base64(image_path)
            if not image_base64:
                logger.error(f"[AuditService] [{sessionID}] ❌ 图片读取失败")
                return R.error(message="图片读取失败", code="500")

            # 检查session是否已有流程状态
            log_step(3, 7, "检查session流程状态", sessionID)
            process_state = get_session_storage_service().get_session_process_state(sessionID)
            is_first_operation = (process_state is None)

            if is_first_operation:
                # ========== 首次操作：识别流程并初始化 ==========
                logger.info(f"[AuditService] [{sessionID}] 📌 首次操作，开始识别流程")
                return await self._handle_first_operation(
                    sessionID, audit_opt, image_base64, image_path, process_name
                )
            else:
                # ========== 后续操作：检查是否在流程中 ==========
                logger.info(f"[AuditService] [{sessionID}] 📌 后续操作")
                return await self._handle_followup_operation(
                    sessionID, audit_opt, image_base64, image_path, process_state
                )

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] ❌ AI审计处理失败: {e}", exc_info=True)
            logger.info("=" * 60)
            return R.error(message="审计处理失败", data=str(e), code="500")

    async def ai_summary(self, request: SummaryRequest) -> R:
        """
        AI总结 - 根据会话中的所有操作记录生成详细工单信息

        Args:
            request: 总结请求对象

        Returns:
            R: 包含工单信息的响应
        """
        try:
            logger.info("=" * 60)
            logger.info(f"[AuditService] [{request.sessionID}] 🔍 开始AI总结")

            # 从数据库获取会话的所有操作记录
            log_step(1, 5, "从数据库获取会话操作记录", request.sessionID)
            records = get_operation_db().get_records_by_session(request.sessionID)

            if not records:
                logger.warning(f"[AuditService] [{request.sessionID}] ⚠️ 未找到操作记录")
                return R.fail(message="未找到操作记录，无法生成工单", code="400")

            logger.info(f"[AuditService] [{request.sessionID}] 找到 {len(records)} 条操作记录")

            # 获取session的流程状态
            log_step(2, 5, "获取session流程状态", request.sessionID)
            process_state = get_session_storage_service().get_session_process_state(request.sessionID)

            # 获取流程信息
            process_info = self._extract_process_info(request.sessionID, process_state)

            # 构建操作摘要文本
            log_step(4, 5, "构建操作摘要", request.sessionID)
            operations_text = self._build_operations_summary(records)

            # 构建流程信息文本
            process_info_text = self._build_process_info_text(process_info)

            # 构建执行情况分析
            execution_analysis = self._build_execution_analysis(process_info)

            # 调用LLM生成工单
            log_step(5, 5, "调用LLM生成详细工单信息", request.sessionID)
            work_order = await self._generate_work_order(
                request.sessionID,
                records,
                operations_text,
                process_info_text,
                execution_analysis
            )

            logger.info(f"[AuditService] [{request.sessionID}] ✅ AI总结完成")
            logger.info(f"[AuditService] [{request.sessionID}] 工单: ds_id={work_order.ds_id}, "
                       f"work_class={work_order.work_class}（{'软件' if work_order.work_class == 1 else '硬件'}）")
            logger.info(f"[AuditService] [{request.sessionID}] 工作内容长度: {len(work_order.work_notice)}字")
            logger.info("=" * 60)

            return R.ok(message="总结完成", data=work_order.model_dump())

        except Exception as e:
            logger.error(f"[AuditService] [{request.sessionID}] ❌ AI总结处理失败: {e}", exc_info=True)
            logger.info("=" * 60)
            return R.error(message="总结处理失败", data=str(e), code="500")

    # ==================== 私有方法：操作处理 ====================

    def _should_ignore_event(self, sessionID: str, audit_opt: AuditOpt) -> bool:
        """
        判断是否应该忽略该事件（事件过滤）

        过滤规则：
        - 鼠标点击事件（ws_mouse_click）：只处理state=true（按下），忽略state=false（松开）
        - 键盘事件（ws_keyboard）：只处理state=true（按下），忽略state=false（松开）
        - 其他事件：不过滤

        Args:
            sessionID: 会话ID
            audit_opt: 审计操作对象

        Returns:
            bool: True表示应该忽略该事件，False表示处理该事件
        """
        try:
            # 尝试解析event_content中的JSON
            event_content_dict = json.loads(audit_opt.event_content)

            # 检查是否包含state字段
            if "state" in event_content_dict:
                state = event_content_dict.get("state")

                # 如果state为false（松开事件），则忽略
                if state is False:
                    logger.debug(f"[AuditService] [{sessionID}] 检测到松开事件: {audit_opt.event_type}")
                    return True

            return False

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # 如果解析失败，不过滤（默认处理）
            logger.debug(f"[AuditService] [{sessionID}] event_content解析失败，不过滤: {e}")
            return False

    def _parse_operation(self, sessionID: str, operation: str) -> AuditOpt:
        """解析operation JSON字符串"""
        try:
            operation_data = json.loads(operation)
            audit_opt = AuditOpt(**operation_data)
            logger.info(f"[AuditService] [{sessionID}] 用户: {audit_opt.user}")
            return audit_opt
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"[AuditService] [{sessionID}] ⚠️ operation解析失败: {e}")
            # 创建一个默认的AuditOpt对象
            return AuditOpt(
                event_time=datetime.now().strftime("%Y%m%d%H%M%S"),
                event_type="unknown",
                event_content=str(operation),
                event_status="UNKNOWN",
                device_id=sessionID,
                device_ip="",
                user="unknown"
            )

    async def _handle_first_operation(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str,
        process_name: Optional[str]
    ) -> R:
        """处理首次操作"""
        # 获取Neo4j连接
        log_step(4, 7, "获取操作流程列表", sessionID)
        yw_neo4j = get_yw_neo4j()
        all_processes = yw_neo4j.get_all_operation_processes()

        if not all_processes:
            logger.warning(f"[AuditService] [{sessionID}] ⚠️ 未找到任何操作流程，使用标准审计流程")
            return await self._standard_audit(sessionID, audit_opt, image_base64, image_path)

        logger.info(f"[AuditService] [{sessionID}] 找到 {len(all_processes)} 个操作流程: {all_processes}")

        # 判断是否使用预设流程
        if process_name:
            logger.info(f"[AuditService] [{sessionID}] 🎯 使用预设流程: {process_name}")
            log_step(5, 7, "使用预设流程", sessionID)

            if process_name not in all_processes:
                logger.warning(f"[AuditService] [{sessionID}] ⚠️ 预设流程'{process_name}'不存在，使用标准审计流程")
                return await self._standard_audit(sessionID, audit_opt, image_base64, image_path)
        else:
            # 调用LLM识别当前操作属于哪个流程
            log_step(5, 7, "调用LLM识别操作流程", sessionID)
            process_name = await self._identify_process(sessionID, audit_opt, image_base64, all_processes)

            if not process_name:
                logger.warning(f"[AuditService] [{sessionID}] ⚠️ LLM无法识别流程，使用标准审计流程")
                return await self._standard_audit(sessionID, audit_opt, image_base64, image_path)

            logger.info(f"[AuditService] [{sessionID}] ✅ 识别为流程: {process_name}")

        # 获取该流程的所有有效操作节点
        log_step(6, 7, "获取流程节点信息", sessionID)
        valid_operations = yw_neo4j.get_operation_process_nodes(process_name)

        if not valid_operations:
            logger.warning(f"[AuditService] [{sessionID}] ⚠️ 流程'{process_name}'没有节点，使用标准审计流程")
            return await self._standard_audit(sessionID, audit_opt, image_base64, image_path)

        logger.info(f"[AuditService] [{sessionID}] 流程包含 {len(valid_operations)} 个有效操作")

        # 保存流程状态到session
        get_session_storage_service().save_session_process_state(
            sessionID=sessionID,
            process_name=process_name,
            valid_operations=valid_operations
        )

        # 执行标准审计（首次操作只检查风险，不检查是否在流程中）
        log_step(7, 7, "执行首次操作审计", sessionID)
        return await self._audit_first_operation(sessionID, audit_opt, image_base64, image_path, process_name)

    async def _handle_followup_operation(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str,
        process_state: Dict
    ) -> R:
        """处理后续操作"""
        process_name = process_state.get("process_name", "未知流程")
        valid_operations = process_state.get("valid_operations", [])
        logger.info(f"[AuditService] [{sessionID}] 📌 当前流程: {process_name}")
        logger.info(f"[AuditService] [{sessionID}] 已执行操作数: {len(process_state.get('current_operations', []))}")

        # 检查操作是否在流程中
        log_step(4, 7, "检查操作是否在流程中", sessionID)
        is_in_process, process_check_result = await self._check_operation_in_process(
            sessionID, audit_opt, image_base64, valid_operations
        )

        # 添加操作到已执行列表
        get_session_storage_service().add_operation_to_session(sessionID, audit_opt.event_content)

        if is_in_process:
            # 在流程中：正常审计
            log_step(5, 7, "操作在流程中，执行标准审计", sessionID)
            result = await self._audit_operation_in_process(sessionID, audit_opt, image_base64, image_path)
            logger.info("=" * 60)
            return result
        else:
            # 跳出流程：根据是否有风险返回不同级别的告警
            log_step(5, 7, "操作跳出流程，评估风险", sessionID)
            return await self._handle_operation_out_of_process(
                sessionID, audit_opt, image_base64, image_path, process_name, process_check_result
            )

    async def _handle_operation_out_of_process(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str,
        process_name: str,
        process_check_result: Dict
    ) -> R:
        """处理跳出流程的操作"""
        has_risk = process_check_result.get("has_risk", False)
        risk_level = process_check_result.get("risk_level", "none")
        reason = process_check_result.get("reason", "操作不在标准流程中")

        # 执行完整的风险审计
        audit_result = await self._audit_operation_risk(sessionID, audit_opt, image_base64)

        # 合并风险判断
        final_has_risk = audit_result.get("has_risk", False) or has_risk
        final_risk_level = self._merge_risk_level(
            audit_result.get("risk_level", "none"),
            risk_level
        )

        # 生成操作总结
        summary = await self._generate_summary(sessionID, audit_opt, image_base64)

        # 保存到数据库
        get_operation_db().save_record(
            session_id=sessionID,
            operation=audit_opt.event_content,
            image_path=image_path,
            summary=summary
        )

        if final_has_risk and final_risk_level in ["high", "medium"]:
            # 严重告警：跳出流程且有风险
            log_step(6, 7, "返回严重告警", sessionID)
            alarm_time = datetime.now()
            result_data = AlarmData(
                equipment_asset=sessionID,
                alarm=f"操作偏离流程'{process_name}'且存在安全风险。{audit_result.get('alarm_message', reason)}",
                alarm_time=alarm_time,
                work_content=summary,
                risk_level=final_risk_level
            )

            logger.warning(f"[AuditService] [{sessionID}] ⚠️ 严重告警: 操作偏离流程且有风险 [{final_risk_level.upper()}]")
            logger.info(f"[AuditService] [{sessionID}] 告警信息: {result_data.alarm}")
            logger.info("=" * 60)

            return R.error(
                message="严重告警：操作偏离流程且存在安全风险",
                code="300001",
                data=result_data.model_dump()
            )
        else:
            # 轻微告警：跳出流程但无显著风险
            log_step(6, 7, "返回轻微告警", sessionID)
            logger.info(f"[AuditService] [{sessionID}] ⚠️ 轻微告警: 操作偏离流程但无显著风险")
            logger.info(f"[AuditService] [{sessionID}] 原因: {reason}")
            logger.info("=" * 60)

            return R.ok(
                message="轻微告警：操作偏离标准流程",
                code="200001",
                data={
                    "equipment_asset": sessionID,
                    "work_content": summary,
                    "process_name": process_name,
                    "reason": reason
                }
            )

    # ==================== 私有方法：LLM调用 ====================

    async def _identify_process(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        all_processes: List[str]
    ) -> Optional[str]:
        """调用LLM识别当前操作属于哪个流程"""
        processes_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(all_processes)])

        system_prompt = """你是一个运维流程识别专家。根据事件类型、事件详情和截图，判断该操作属于哪个预定义的运维流程。

【判断要点】
- event_type 告诉你这是什么类型的事件（如鼠标点击、键盘输入、系统命令等）
- event_content 提供事件的技术细节（如位置、按键、参数等）
- 结合截图内容，综合判断这属于哪个运维流程

【输出要求】
必须严格返回JSON格式（不要使用markdown代码块）：
{
  "process_name": "流程名称（必须从提供的流程列表中选择）",
  "confidence": "high/medium/low",
  "reason": "选择理由（20-50字）"
}

注意：
- process_name必须完全匹配提供的流程名称之一
- 如果无法确定，选择confidence为"low"的最可能流程
- 如果完全无法匹配，返回null作为process_name"""

        user_prompt = f"""请判断以下操作属于哪个运维流程：

事件类型：{audit_opt.event_type}
事件详情：{audit_opt.event_content}

可用的运维流程列表：
{processes_text}

请结合截图内容和事件类型，从上述流程列表中选择最匹配的一个，并严格按JSON格式返回结果。"""

        try:
            response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_base64,
                temperature=0.1,
                max_tokens=500,
                system_prompt=system_prompt
            )

            if not response:
                return None

            result = self._parse_json_response(response)
            if not result:
                return None

            process_name = result.get("process_name")
            confidence = result.get("confidence", "low")
            reason = result.get("reason", "")

            if not process_name or process_name == "null":
                logger.info(f"[AuditService] [{sessionID}] LLM无法识别流程: {reason}")
                return None

            # 验证流程名称是否在列表中
            if process_name not in all_processes:
                logger.warning(f"[AuditService] [{sessionID}] LLM返回的流程'{process_name}'不在列表中，尝试模糊匹配")
                for p in all_processes:
                    if process_name in p or p in process_name:
                        logger.info(f"[AuditService] [{sessionID}] 模糊匹配到: {p}")
                        return p
                return None

            logger.info(f"[AuditService] [{sessionID}] LLM识别结果: {process_name} (置信度: {confidence}, 理由: {reason})")
            return process_name

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 流程识别失败: {e}")
            return None

    async def _check_operation_in_process(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        valid_operations: List[str]
    ) -> Tuple[bool, Dict]:
        """检查操作是否在流程中（结合历史操作上下文）"""
        operations_text = "\n".join([f"- {op}" for op in valid_operations])

        event_content_display = self._format_event_content(audit_opt.event_content)

        # ========== 新增：获取历史操作上下文 ==========
        history_summary = self._get_history_summary(sessionID)

        system_prompt = """你是一个运维流程检查专家。判断当前操作是否属于给定的标准操作流程。

【重要规则：图形界面文件操作全部告警】
在合格的运维场景中，通过截图识别出以下文件操作时必须触发告警：

**图形界面的文件操作（鼠标操作）：**
- **打开文件/文件夹**：
  - 双击文件或文件夹
  - 右键选择"打开"
  - 点击"打开"按钮（如文件对话框中）
  - 在文件管理器中点击文件名
  - 在应用程序中点击"文件→打开"

- **删除文件/文件夹**：
  - 右键选择"删除"
  - 拖动到回收站/垃圾箱
  - 选中后按Delete键
  - 点击"删除"按钮

- **移动文件/文件夹**：
  - 鼠标拖动文件到其他文件夹
  - 剪切粘贴（Ctrl+X → Ctrl+V）
  - 右键"剪切"后粘贴

- **复制文件/文件夹**：
  - 右键"复制"
  - 拖动复制（Ctrl+拖动）
  - Ctrl+C → Ctrl+V

- **重命名文件/文件夹**：
  - 右键选择"重命名"
  - 选中后按F2键

- **其他文件操作**：
  - 显示文件管理器窗口（Windows资源管理器、Linux文件管理器、macOS Finder等）
  - 显示文件选择对话框（打开/保存对话框）
  - 显示文件内容编辑界面
  - 右键菜单中出现文件相关选项

**命令行的文件操作（如果终端可见）：**
- rm、cp、mv、vi、vim、nano、cat、less、more、head、tail等文件操作命令

上述文件操作无论是否在标准流程中，都必须标记为有风险（has_risk=true, risk_level="high"）。

【判断标准】
- 事件类型和操作内容与流程中的某个步骤匹配，则认为在流程中
- 操作的目标、对象、动作与流程中的某个步骤一致，则认为在流程中
- 虽然表述不同但本质相同的操作，应认为在流程中
- event_type可以帮助快速判断操作类型（鼠标点击、键盘输入、系统命令等）
- 结合历史操作上下文，判断当前操作的合理性和连贯性
- **即使操作在流程中，如果涉及文件操作也必须标记风险**
- **重点分析截图内容：识别文件管理器窗口、文件对话框、右键菜单、拖动操作等**

【输出要求】
必须严格返回JSON格式（不要使用markdown代码块）：
{
  "is_in_process": true或false,
  "matched_operation": "匹配的流程操作名称（如果不完全匹配，填最接近的）",
  "has_risk": true或false,
  "risk_level": "high/medium/low/none",
  "reason": "判断理由（30-80字）"
}

注意：
- is_in_process为true时，matched_operation填写匹配的步骤
- is_in_process为false时，matched_operation填写"无匹配"或最接近的步骤
- 如果操作不在流程中，评估该偏离操作本身是否有风险
- 历史操作上下文可以帮助判断当前操作是否合理
- **任何涉及文件的操作（图形界面或命令行）都必须告警**，这是运维安全的强制要求
- **必须仔细分析截图内容**，识别文件管理器、对话框、右键菜单等GUI元素"""

        user_prompt = f"""请判断以下操作是否在标准运维流程中：

事件类型：{audit_opt.event_type}
事件详情：{event_content_display}

标准流程包含的所有操作步骤：
{operations_text}
{history_summary}

请结合截图内容、事件类型和历史操作上下文进行判断，并严格按JSON格式返回结果。"""

        try:
            response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_base64,
                temperature=0.1,
                max_tokens=500,
                system_prompt=system_prompt
            )

            if not response:
                return True, {"has_risk": False, "risk_level": "none", "reason": "LLM调用失败，默认认为在流程中"}

            result = self._parse_json_response(response)
            if not result:
                return True, {"has_risk": False, "risk_level": "none", "reason": "解析失败，默认认为在流程中"}

            is_in_process = result.get("is_in_process", False)
            matched_operation = result.get("matched_operation", "")
            has_risk = result.get("has_risk", False)
            risk_level = result.get("risk_level", "none")
            reason = result.get("reason", "")

            logger.info(f"[AuditService] [{sessionID}] 流程检查: {'✓在流程中' if is_in_process else '✗跳出流程'}")
            logger.info(f"[AuditService] [{sessionID}] 匹配操作: {matched_operation}, 风险: {risk_level}")

            return is_in_process, {
                "matched_operation": matched_operation,
                "has_risk": has_risk,
                "risk_level": risk_level,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 流程检查失败: {e}")
            return True, {"has_risk": False, "risk_level": "none", "reason": "流程检查异常"}

    async def _audit_operation_risk(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str
    ) -> Dict:
        """审计操作的风险（结合历史操作上下文）"""
        event_content_display = self._format_event_content(audit_opt.event_content)

        # ========== 新增：获取历史操作上下文 ==========
        history_summary = self._get_history_summary(sessionID)

        system_prompt = """你是一个专业的运维安全审计AI助手。判断运维操作是否存在安全风险。

【重要规则：图形界面文件操作全部告警】
在合格的运维场景中，通过截图识别出以下文件操作时必须触发告警：

**图形界面的文件操作（鼠标操作）：**
- **打开文件/文件夹**：
  - 双击文件或文件夹
  - 右键选择"打开"
  - 点击"打开"按钮（如文件对话框中）
  - 在文件管理器中点击文件名
  - 在应用程序中点击"文件→打开"

- **删除文件/文件夹**：
  - 右键选择"删除"
  - 拖动到回收站/垃圾箱
  - 选中后按Delete键
  - 点击"删除"按钮

- **移动文件/文件夹**：
  - 鼠标拖动文件到其他文件夹
  - 剪切粘贴（Ctrl+X → Ctrl+V）
  - 右键"剪切"后粘贴

- **复制文件/文件夹**：
  - 右键"复制"
  - 拖动复制（Ctrl+拖动）
  - Ctrl+C → Ctrl+V

- **重命名文件/文件夹**：
  - 右键选择"重命名"
  - 选中后按F2键

- **其他文件操作**：
  - 显示文件管理器窗口（Windows资源管理器、Linux文件管理器、macOS Finder等）
  - 显示文件选择对话框（打开/保存对话框）
  - 显示文件内容编辑界面
  - 右键菜单中出现文件相关选项

**命令行的文件操作（如果终端可见）：**
- rm、cp、mv、vi、vim、nano、cat、less、more、head、tail等文件操作命令

上述文件操作无论是否在标准流程中，都必须标记为有风险（has_risk=true, risk_level="high"）。

【判断标准】
**高危操作（has_risk=true, risk_level="high"）**：
- 所有文件操作（打开、删除、移动、复制、重命名等）
- 删除关键数据、格式化磁盘、停止核心服务
- 修改生产环境核心配置（防火墙、数据库、网络、系统配置）
- 执行未知来源的脚本或命令
- 危险命令：rm -rf /、dd、shutdown、format等
- 在生产环境执行非授权的修改操作
- 结合历史操作，发现异常或重复的危险操作

**中危操作（has_risk=true, risk_level="medium"）**：
- 修改非关键配置
- 重启非核心服务
- 可能影响性能的操作
- 操作流程不规范但未造成直接影响

**低危操作（has_risk=true, risk_level="low"）**：
- 轻微操作不规范
- 操作顺序有误但无安全影响

**安全操作（has_risk=false, risk_level="none"）**：
- 仅查询类操作（如查看进程列表、系统信息等，不涉及文件内容读取）
- 符合流程的标准操作（且不涉及文件操作）
- 结合历史操作，属于正常流程的一部分（且不涉及文件操作）

【输出要求】
必须严格返回JSON格式：
{
  "has_risk": true或false,
  "risk_level": "high/medium/low/none",
  "alarm_message": "具体告警内容（20-100字）"
}

注意：
- 历史操作上下文可以帮助判断当前操作是否异常或重复
- 某些操作单独看安全，但结合历史可能有风险
- **任何涉及文件的操作（图形界面或命令行）都必须告警**，这是运维安全的强制要求
- **必须仔细分析截图内容**，识别文件管理器、对话框、右键菜单、拖动操作等GUI元素"""

        user_prompt = f"""请审计以下运维操作：

事件类型：{audit_opt.event_type}
事件详情：{event_content_display}
{history_summary}

请结合截图内容、事件类型和历史操作上下文判断风险等级，并严格按JSON格式返回结果。"""

        try:
            response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_base64,
                temperature=0.1,
                max_tokens=500,
                system_prompt=system_prompt
            )

            if not response:
                return {"has_risk": False, "risk_level": "none", "alarm_message": ""}

            result = self._parse_json_response(response)
            if not result:
                return {"has_risk": False, "risk_level": "none", "alarm_message": ""}

            return {
                "has_risk": result.get("has_risk", False),
                "risk_level": result.get("risk_level", "none"),
                "alarm_message": result.get("alarm_message", "")
            }

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 风险审计失败: {e}")
            # 解析失败时的保守策略
            error_keywords = ["删除", "格式化", "shutdown", "rm -rf", "drop", "truncate"]
            has_risk = any(keyword in audit_opt.event_content.lower() for keyword in error_keywords)
            return {
                "has_risk": has_risk,
                "risk_level": "medium" if has_risk else "none",
                "alarm_message": "检测到可能的危险操作" if has_risk else ""
            }

    async def _generate_summary(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str
    ) -> str:
        """生成操作总结"""
        try:
            event_content_display = self._format_event_content(audit_opt.event_content)

            # 根据event_type生成不同的总结描述
            if audit_opt.event_type == "ws_mouse_click":
                operation_desc = f"鼠标点击操作，详情：{event_content_display}"
            elif audit_opt.event_type == "ws_keyboard":
                operation_desc = f"键盘输入操作，详情：{event_content_display}"
            elif "command" in audit_opt.event_type.lower():
                operation_desc = f"命令执行操作，详情：{event_content_display}"
            else:
                operation_desc = f"{audit_opt.event_type}操作，详情：{event_content_display}"

            summary_prompt = f"请用一句话（30字内）概括这个操作：{operation_desc}"
            summary = llm_client.chat_with_vision(
                prompt=summary_prompt,
                image_base64=image_base64,
                temperature=0.1,
                max_tokens=200,
                system_prompt="你是一个运维操作记录助手，请简洁概括操作内容。"
            )
            return summary or operation_desc
        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 生成总结失败: {e}")
            return f"{audit_opt.event_type}: {audit_opt.event_content}"

    # ==================== 私有方法：审计流程 ====================

    async def _standard_audit(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str
    ) -> R:
        """标准审计流程（没有流程配置时的降级处理）"""
        audit_result = await self._audit_operation_risk(sessionID, audit_opt, image_base64)
        summary = await self._generate_summary(sessionID, audit_opt, image_base64)

        # 保存到数据库
        get_operation_db().save_record(
            session_id=sessionID,
            operation=audit_opt.event_content,
            image_path=image_path,
            summary=summary
        )

        if audit_result.get("has_risk") and audit_result.get("risk_level") != "none":
            alarm_time = datetime.now()
            result_data = AlarmData(
                equipment_asset=sessionID,
                alarm=audit_result.get("alarm_message", "检测到安全风险"),
                alarm_time=alarm_time,
                work_content=summary,
                risk_level=audit_result.get("risk_level", "medium")
            )

            logger.warning(f"[AuditService] [{sessionID}] ⚠️ 标准审计发现风险")
            return R.error(
                message="发现安全风险",
                code="300001",
                data=result_data.model_dump()
            )
        else:
            return R.ok(
                message="操作正常",
                data={"equipment_asset": sessionID, "work_content": summary}
            )

    async def _audit_first_operation(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str,
        process_name: str
    ) -> R:
        """首次操作的审计"""
        audit_result = await self._audit_operation_risk(sessionID, audit_opt, image_base64)
        summary = await self._generate_summary(sessionID, audit_opt, image_base64)

        # 添加操作到已执行列表
        get_session_storage_service().add_operation_to_session(sessionID, audit_opt.event_content)

        # 保存到数据库
        get_operation_db().save_record(
            session_id=sessionID,
            operation=audit_opt.event_content,
            image_path=image_path,
            summary=summary
        )

        if audit_result.get("has_risk") and audit_result.get("risk_level") != "none":
            alarm_time = datetime.now()
            result_data = AlarmData(
                equipment_asset=sessionID,
                alarm=audit_result.get("alarm_message", "检测到安全风险"),
                alarm_time=alarm_time,
                work_content=summary,
                risk_level=audit_result.get("risk_level", "medium")
            )

            logger.info(f"[AuditService] [{sessionID}] ⚠️ 首次操作发现风险")
            logger.info("=" * 60)

            return R.error(
                message=f"已识别流程'{process_name}'，但操作存在风险",
                code="300001",
                data=result_data.model_dump()
            )
        else:
            logger.info(f"[AuditService] [{sessionID}] ✅ 首次操作正常，流程: {process_name}")
            logger.info("=" * 60)

            return R.ok(
                message=f"已识别流程: {process_name}",
                data={"equipment_asset": sessionID, "work_content": summary, "process_name": process_name}
            )

    async def _audit_operation_in_process(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str
    ) -> R:
        """操作在流程中的审计"""
        audit_result = await self._audit_operation_risk(sessionID, audit_opt, image_base64)
        summary = await self._generate_summary(sessionID, audit_opt, image_base64)

        # 保存到数据库
        get_operation_db().save_record(
            session_id=sessionID,
            operation=audit_opt.event_content,
            image_path=image_path,
            summary=summary
        )

        if audit_result.get("has_risk") and audit_result.get("risk_level") != "none":
            alarm_time = datetime.now()
            result_data = AlarmData(
                equipment_asset=sessionID,
                alarm=audit_result.get("alarm_message", "检测到安全风险"),
                alarm_time=alarm_time,
                work_content=summary,
                risk_level=audit_result.get("risk_level", "medium")
            )

            logger.warning(f"[AuditService] [{sessionID}] ⚠️ 操作在流程中但存在风险")

            return R.error(
                message="操作在流程中，但存在安全风险",
                code="300001",
                data=result_data.model_dump()
            )
        else:
            logger.info(f"[AuditService] [{sessionID}] ✅ 操作正常，在流程中")
            return R.ok(
                message="操作正常",
                data={"equipment_asset": sessionID, "work_content": summary}
            )

    # ==================== 私有方法：工单生成 ====================

    def _extract_process_info(self, sessionID: str, process_state: Optional[Dict]) -> Dict:
        """提取流程信息"""
        process_name = None
        process_chain = []
        valid_operations = []
        executed_operations = []

        if process_state:
            process_name = process_state.get('process_name')
            valid_operations = process_state.get('valid_operations', [])
            executed_operations = process_state.get('current_operations', [])
            logger.info(f"[AuditService] [{sessionID}] 流程名称: {process_name}")
            logger.info(f"[AuditService] [{sessionID}] 流程节点数: {len(valid_operations)}")
            logger.info(f"[AuditService] [{sessionID}] 已执行操作数: {len(executed_operations)}")

            # 从Neo4j获取流程链条信息
            if process_name:
                log_step(3, 5, "获取流程链条信息", sessionID)
                yw_neo4j = get_yw_neo4j()
                process_chain = yw_neo4j.get_operation_process_chain(process_name)
                logger.info(f"[AuditService] [{sessionID}] 流程关系数: {len(process_chain)}")
        else:
            logger.info(f"[AuditService] [{sessionID}] 无流程状态记录")

        return {
            "process_name": process_name,
            "process_chain": process_chain,
            "valid_operations": valid_operations,
            "executed_operations": executed_operations
        }

    def _build_operations_summary(self, records: List[Dict]) -> str:
        """构建操作摘要文本"""
        operations_summary = []
        for idx, record in enumerate(records, 1):
            operations_summary.append(
                f"操作{idx}:\n"
                f"- 操作描述: {record['operation']}\n"
                f"- AI总结: {record['summary']}\n"
                f"- 时间: {record['created_at']}"
            )
        return "\n\n".join(operations_summary)

    def _build_process_info_text(self, process_info: Dict) -> str:
        """构建流程信息文本"""
        process_name = process_info.get("process_name")
        valid_operations = process_info.get("valid_operations", [])
        process_chain = process_info.get("process_chain", [])

        if process_name and valid_operations:
            text = f"""
标准流程参考：{process_name}
流程包含的步骤：
{chr(10).join([f'{i+1}. {op}' for i, op in enumerate(valid_operations)])}
"""
            if process_chain:
                text += "\n流程执行顺序：\n"
                for idx, chain in enumerate(process_chain, 1):
                    text += f"{idx}. {chain['from']} → {chain['to']}\n"
            return text
        else:
            return "无标准流程参考（自由操作模式）"

    def _build_execution_analysis(self, process_info: Dict) -> str:
        """构建执行情况分析"""
        process_name = process_info.get("process_name")
        valid_operations = process_info.get("valid_operations", [])
        executed_operations = process_info.get("executed_operations", [])

        if process_name and executed_operations:
            return f"""
流程执行情况：
- 标准流程步骤数: {len(valid_operations)}
- 实际执行操作数: {len(executed_operations)}
- 执行进度: {len(executed_operations)}/{len(valid_operations)}
"""
        return ""

    async def _generate_work_order(
        self,
        sessionID: str,
        records: List[Dict],
        operations_text: str,
        process_info_text: str,
        execution_analysis: str
    ) -> WorkOrderData:
        """生成工单"""
        system_prompt = """你是一个专业的运维工单生成AI助手。你的任务是根据运维操作记录和标准流程信息生成详细的工单信息。

工单分类说明（work_class）：
- 1: 软件（涉及软件安装、配置、调试、升级等）
- 2: 硬件（涉及硬件设备维护、更换、维修等）

**工作内容（work_notice）要求：**
- 必须详细描述所有操作步骤
- 如果在标准流程中，说明执行的步骤和进度
- 如果偏离流程，说明具体偏离情况
- 包含具体的设备、软件、配置信息
- 说明操作目的和结果
- 字数要求：至少150字，确保信息完整

响应格式要求（必须是JSON格式）：
{
    "work_class": 工单分类（整数，1=软件，2=硬件）,
    "work_notice": "详细的工作内容总结（至少150字，包含所有操作细节）"
}

请始终返回有效的JSON格式。"""

        user_prompt = f"""请根据以下运维操作记录和流程信息生成详细的工单信息：

会话ID（设备ID）: {sessionID}
共有 {len(records)} 条操作记录。

{process_info_text}

{execution_analysis}

实际操作记录详情（AI已总结每个操作的关键内容）：
{operations_text}

请综合分析：
1. 判断主要是软件操作还是硬件操作
2. **生成详细的工作内容总结（至少150字）**，包括：
   - 所有操作步骤
   - 涉及的设备和组件
   - 配置修改内容
   - 操作目的和结果
   - 如果在标准流程中，说明执行情况和进度

请按照要求的JSON格式返回结果。"""

        # 调用LLM进行总结
        llm_response = llm_client.chat_with_siliconflow(
            prompt=user_prompt,
            temperature=0.3,
            max_tokens=2000,
            system_prompt=system_prompt
        )

        if not llm_response:
            logger.error(f"[AuditService] [{sessionID}] ❌ LLM调用失败")
            # 返回默认工单
            return WorkOrderData(
                ds_id=sessionID,
                work_class=1,
                work_notice="AI分析失败，无法生成详细工单"
            )

        # 解析LLM响应
        try:
            result = self._parse_json_response(llm_response)
            if result:
                return WorkOrderData(
                    ds_id=sessionID,
                    work_class=int(result.get("work_class", 1)),
                    work_notice=result.get("work_notice", "运维操作总结")
                )
        except Exception as e:
            logger.warning(f"[AuditService] [{sessionID}] ⚠️ JSON解析失败: {e}")

        # 如果解析失败，使用默认值
        basic_summary = "运维操作包括：" + "；".join([r['summary'] for r in records[:5]])
        return WorkOrderData(
            ds_id=sessionID,
            work_class=1,
            work_notice=basic_summary
        )

    # ==================== 私有方法：工具函数 ====================

    def _get_history_summary(self, sessionID: str, max_operations: int = 10) -> str:
        """
        获取历史操作总结（用于LLM上下文）

        Args:
            sessionID: 会话ID
            max_operations: 最多返回的历史操作数量（默认10条，避免token过多）

        Returns:
            历史操作总结文本
        """
        try:
            # 从数据库获取该session的所有历史操作记录
            records = get_operation_db().get_records_by_session(sessionID)

            if not records:
                return "\n历史操作：无历史操作记录"

            # 限制历史操作数量，避免上下文过长
            recent_records = records[-max_operations:] if len(records) > max_operations else records

            # 构建历史操作摘要
            history_items = []
            for idx, record in enumerate(recent_records, 1):
                history_items.append(
                    f"历史操作{idx}: {record['summary']}"
                )

            history_text = "\n".join(history_items)

            return f"\n\n【历史操作上下文（最近{len(recent_records)}条）】\n{history_text}"

        except Exception as e:
            logger.warning(f"[AuditService] [{sessionID}] 获取历史操作失败: {e}")
            return "\n历史操作：无法获取历史记录"

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """解析LLM返回的JSON"""
        try:
            response_text = response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            return json.loads(response_text)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"[AuditService] JSON解析失败: {e}")
            return None

    def _format_event_content(self, event_content: str) -> str:
        """格式化事件内容"""
        try:
            event_content_detail = json.loads(event_content)
            return json.dumps(event_content_detail, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            return event_content

    def _merge_risk_level(self, level1: str, level2: str) -> str:
        """合并两个风险等级，返回更高的"""
        risk_order = ["none", "low", "medium", "high"]
        idx1 = risk_order.index(level1) if level1 in risk_order else 0
        idx2 = risk_order.index(level2) if level2 in risk_order else 0
        return risk_order[max(idx1, idx2)]


# 全局服务实例（延迟初始化）
_audit_service_instance: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """获取AuditService单例"""
    global _audit_service_instance
    if _audit_service_instance is None:
        _audit_service_instance = AuditService()
    return _audit_service_instance
