"""
AI审计服务（简化版）- 完全基于LLM的风险审计，使用MySQL存储操作记录
"""
import json
from datetime import datetime
from typing import Optional, Dict

from models.schemas import R, AuditOpt, AlarmData
from core.llm_client import llm_client
from core.mysql_db import get_operation_db
from services.session_storage_service import get_session_storage_service
from utils.logger import logger, log_step


def _build_response_data(sessionID: str, alarm: Optional[str] = None, alarm_time: Optional[datetime] = None) -> dict:
    """
    构建统一格式的响应数据

    Args:
        sessionID: 会话ID（设备资产编号）
        alarm: 告警信息（可选，无告警时为None）
        alarm_time: 告警时间（可选，无告警时为None）

    Returns:
        dict: 包含3个字段的响应数据
    """
    return {
        "equipment_asset": sessionID,
        "alarm": alarm,  # 无告警时为None（JSON中会变成null）
        "alarm_time": alarm_time  # 无告警时为None（JSON中会变成null）
    }


class AuditService:
    """AI审计服务（纯LLM版本，无数据库依赖）"""

    def __init__(self):
        """初始化服务"""
        logger.info("✅ AuditService 初始化完成（纯LLM版本，无数据库依赖）")

    async def ai_check(
        self,
        pic_filename: str,
        image_data: bytes,
        sessionID: str,
        operation: str,
        process_name: Optional[str] = None
    ) -> R:
        """
        AI审计 - 基于LLM的智能风险审计（使用MySQL存储操作记录）

        Args:
            pic_filename: 图片文件名
            image_data: 图片数据
            sessionID: 会话ID（设备ID）
            operation: 图片对应的操作描述（JSON字符串）
            process_name: 预设流程名称（可选，仅用于记录，不影响审计逻辑）

        Returns:
            R: 审计结果
            - code="200": 操作正常（无风险）
            - code="200001": 轻微告警（可选，用于未来扩展）
            - code="300001": 严重告警（存在安全风险）
        """
        try:
            logger.info("=" * 60)
            logger.info(f"[AuditService] [{sessionID}] 🔍 开始AI审计（纯LLM模式）")

            # 解析operation JSON字符串
            audit_opt = self._parse_operation(sessionID, operation)
            logger.info(f"[AuditService] [{sessionID}] 事件类型: {audit_opt.event_type}")

            # ========== 事件过滤：只处理"按下"事件 ==========
            if self._should_ignore_event(sessionID, audit_opt):
                logger.info(f"[AuditService] [{sessionID}] ⏭️ 忽略松开事件（state=false）")
                logger.info("=" * 60)
                return R.ok(message="忽略松开事件", data=_build_response_data(sessionID))

            # 保存图片到会话目录
            log_step(1, 4, "保存图片", sessionID)
            filename = pic_filename or f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = get_session_storage_service().save_image(sessionID, filename, image_data)

            # 读取图片并转换为base64
            log_step(2, 4, "准备图片数据", sessionID)
            image_base64 = get_session_storage_service().get_image_base64(image_path)
            if not image_base64:
                logger.error(f"[AuditService] [{sessionID}] ❌ 图片读取失败")
                return R.error(message="图片读取失败", code="500", data=_build_response_data(sessionID))

            # 执行纯LLM风险审计（包含数据库保存）
            return await self._llm_risk_audit(
                sessionID=sessionID,
                audit_opt=audit_opt,
                image_base64=image_base64,
                image_path=image_path,
                process_name=process_name
            )

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] ❌ AI审计处理失败: {e}", exc_info=True)
            logger.info("=" * 60)
            return R.error(message="审计处理失败", code="500", data=_build_response_data(sessionID))

    # ==================== 私有方法：基础功能 ====================

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

    # ==================== 私有方法：核心审计逻辑 ====================

    async def _llm_risk_audit(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str,
        image_path: str,
        process_name: Optional[str] = None
    ) -> R:
        """
        纯LLM风险审计（核心逻辑）

        直接使用LLM判断：
        1. 当前操作是否有风险
        2. 风险等级（high/medium/low/none）
        3. 生成告警消息（如果有风险）
        4. 生成操作总结
        5. 保存操作记录到MySQL数据库
        """
        log_step(3, 4, "LLM智能风险审计", sessionID)

        # 调用LLM进行风险判断
        audit_result = await self._audit_operation_risk(sessionID, audit_opt, image_base64)

        # 生成操作总结
        summary = await self._generate_summary(sessionID, audit_opt, image_base64)

        # 保存操作记录到数据库
        log_step(4, 4, "保存操作记录到数据库", sessionID)
        get_operation_db().save_record(
            session_id=sessionID,
            operation=audit_opt.event_content,
            image_path=image_path,
            summary=summary
        )

        # 根据风险等级返回不同结果
        has_risk = audit_result.get("has_risk", False)
        risk_level = audit_result.get("risk_level", "none")

        if has_risk and risk_level != "none":
            # 有风险：返回告警
            alarm_time = datetime.now()
            result_data = AlarmData(
                equipment_asset=sessionID,
                alarm=audit_result.get("alarm_message", "检测到安全风险"),
                alarm_time=alarm_time,
                work_content=summary,
                risk_level=risk_level
            )

            logger.warning(f"[AuditService] [{sessionID}] ⚠️ 发现安全风险 [{risk_level.upper()}]")
            logger.info(f"[AuditService] [{sessionID}] 告警信息: {result_data.alarm}")
            logger.info("=" * 60)

            return R.fail(
                message="发现安全风险",
                code="300001",
                data=result_data.to_api_response()
            )
        else:
            # 无风险：返回正常
            logger.info(f"[AuditService] [{sessionID}] ✅ 操作正常，无风险")

            # 如果有流程名称，附加到返回信息
            message = "操作正常"
            if process_name:
                message = f"已识别流程: {process_name}"
                logger.info(f"[AuditService] [{sessionID}] 流程: {process_name}")

            logger.info("=" * 60)

            return R.ok(
                message=message,
                data=_build_response_data(sessionID)  # 无告警，alarm和alarm_time为None
            )

    # ==================== 私有方法：LLM调用 ====================

    async def _audit_operation_risk(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str
    ) -> Dict:
        """
        使用LLM审计操作的风险

        这是核心的审计逻辑，完全依赖LLM的视觉理解和推理能力
        """
        event_content_display = self._format_event_content(audit_opt.event_content)

        # 获取最近的操作历史作为上下文（最近20条操作）
        recent_records = get_operation_db().get_records_by_session(sessionID)
        recent_operations = []
        for record in recent_records[-20:]:  # 取最近20条
            try:
                op_data = json.loads(record['operation'])
                recent_operations.append({
                    'event_type': op_data.get('event_type', 'unknown'),
                    'event_time': op_data.get('event_time', ''),
                    'summary': record.get('summary', '')
                })
            except:
                pass

        # 构建历史上下文
        history_context = ""
        if recent_operations:
            history_context = "\n\n【最近操作历史】\n"
            for i, op in enumerate(recent_operations[-10:], 1):  # 只显示最近10条
                history_context += f"{i}. {op['event_type']} - {op['summary']}\n"

        # 精简版系统提示词，针对 flash 模型优化
        system_prompt = """你是运维安全审计AI，判断操作是否有风险。

【正常流程（不报错）】
1. 系统重装：点击安装界面、语言选择、磁盘配置、Root密码设置、开始安装、许可协议
2. 密码重置：打开终端、输入passwd命令、输入密码、重启命令

【风险判断】
🔴 高危（risk_level="high"）：
- 访问 /root、/etc、/boot、/sys、/proc
- 打开 /etc/passwd、/etc/shadow、私钥文件
- 删除、格式化、停止核心服务
- 修改用户权限、创建管理员账户

🟡 中危（risk_level="medium"）：
- 打开配置文件（*.conf、*.cfg、.env、*.yaml）
- 打开数据库（*.sql、*.db）
- 打开脚本文件（*.sh、*.py、*.js）
- 访问 /home、~、用户目录
- 安装新软件

🟢 低危（risk_level="low"）：
- 打开普通文件夹（/tmp、/opt）
- 浏览文件系统
- 查看日志文件

✅ 安全（risk_level="none"）：
- 点击应用按钮、菜单
- 应用内部操作（不涉及文件访问）

【关键规则】
- 必须识别截图中的文件路径！
- 打开文件夹/文件 = 有风险（除非是正常流程）
- 根据操作历史判断是否在执行流程

【输出格式】纯JSON（不要```）：
{"has_risk": true/false, "risk_level": "high/medium/low/none", "alarm_message": "原因（20-100字）"}"""

        user_prompt = f"""请审计以下运维操作：
{history_context}

【当前待审计的操作】
事件类型：{audit_opt.event_type}
事件详情：{event_content_display}

请结合截图内容、事件类型和最近的操作历史，判断该操作是否存在安全风险，并严格按JSON格式返回结果。

特别注意：
1. 先根据历史操作判断当前是否正在执行系统重装或密码重置流程
2. 如果正在执行这些流程且当前操作是流程的正常步骤，应判定为安全操作（has_risk=false, risk_level="none"）
3. 如果明显偏离流程或存在其他安全风险，才报告风险"""

        try:
            # 针对 flash 模型优化：提高温度以增加推理多样性，增加 token 限制
            response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_base64,
                temperature=0.3,  # 从 0.1 提高到 0.3，让 flash 模型更有可能输出风险判断
                max_tokens=800,   # 从 500 提高到 800，确保有足够空间输出详细判断
                system_prompt=system_prompt
            )

            if not response:
                logger.warning(f"[AuditService] [{sessionID}] LLM未返回响应，使用默认安全值")
                return {"has_risk": False, "risk_level": "none", "alarm_message": ""}

            result = self._parse_json_response(response)
            if not result:
                logger.warning(f"[AuditService] [{sessionID}] JSON解析失败，使用默认安全值")
                return {"has_risk": False, "risk_level": "none", "alarm_message": ""}

            return {
                "has_risk": result.get("has_risk", False),
                "risk_level": result.get("risk_level", "none"),
                "alarm_message": result.get("alarm_message", "")
            }

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 风险审计失败: {e}")
            # 解析失败时的保守策略：使用关键词匹配
            error_keywords = [
                "删除", "delete", "drop", "truncate",
                "格式化", "format", "rm -rf",
                "shutdown", "停止", "stop"
            ]
            event_text = audit_opt.event_content.lower()
            has_risk = any(keyword.lower() in event_text for keyword in error_keywords)

            if has_risk:
                return {
                    "has_risk": True,
                    "risk_level": "medium",
                    "alarm_message": "检测到可能的危险操作关键词"
                }
            else:
                return {
                    "has_risk": False,
                    "risk_level": "none",
                    "alarm_message": ""
                }

    async def _generate_summary(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str
    ) -> str:
        """
        生成操作总结（简洁版）

        用一句话概括当前操作的内容
        """
        try:
            event_content_display = self._format_event_content(audit_opt.event_content)

            # 根据event_type生成不同的总结描述
            if audit_opt.event_type == "ws_mouse_click":
                operation_desc = f"鼠标点击操作，坐标信息：{event_content_display}"
            elif audit_opt.event_type == "ws_keyboard":
                operation_desc = f"键盘输入操作，按键信息：{event_content_display}"
            elif "command" in audit_opt.event_type.lower():
                operation_desc = f"命令执行操作：{event_content_display}"
            else:
                operation_desc = f"{audit_opt.event_type}操作：{event_content_display}"

            summary_prompt = f"""请用一句话（30字以内）概括这个运维操作：

{operation_desc}

要求：简洁明了，说明操作类型和主要目标。"""

            summary = llm_client.chat_with_vision(
                prompt=summary_prompt,
                image_base64=image_base64,
                temperature=0.3,
                max_tokens=150,
                system_prompt="你是一个运维操作记录助手，擅长简洁概括操作内容。"
            )

            # 如果LLM返回为空，使用默认描述
            if not summary or len(summary.strip()) == 0:
                if audit_opt.event_type == "ws_mouse_click":
                    return "鼠标点击操作"
                elif audit_opt.event_type == "ws_keyboard":
                    return "键盘输入操作"
                else:
                    return f"{audit_opt.event_type}操作"

            return summary.strip()

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 生成总结失败: {e}")
            # 返回基础描述
            if audit_opt.event_type == "ws_mouse_click":
                return "鼠标点击操作"
            elif audit_opt.event_type == "ws_keyboard":
                return "键盘输入操作"
            else:
                return f"{audit_opt.event_type}操作"

    # ==================== 私有方法：工具函数 ====================

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        解析LLM返回的JSON

        容错处理：
        1. 自动移除markdown代码块标记（```json 和 ```）
        2. 处理前后空白字符
        3. 智能提取JSON对象（通过定位{和}）
        """
        try:
            response_text = response.strip()

            # 方法1: 智能提取 - 直接定位JSON对象的开始和结束
            # 找到第一个 { 或 [ 的位置
            json_start = -1
            for i, char in enumerate(response_text):
                if char in ['{', '[']:
                    json_start = i
                    break

            if json_start == -1:
                logger.warning(f"[AuditService] 未找到JSON开始标记，原始响应前200字符: {response[:200]}")
                return None

            # 找到最后一个 } 或 ] 的位置
            json_end = -1
            for i in range(len(response_text) - 1, -1, -1):
                if response_text[i] in ['}', ']']:
                    json_end = i
                    break

            if json_end == -1 or json_end <= json_start:
                logger.warning(f"[AuditService] 未找到JSON结束标记，原始响应前200字符: {response[:200]}")
                return None

            # 提取JSON内容
            json_content = response_text[json_start:json_end + 1]

            # 尝试解析
            result = json.loads(json_content)

            logger.debug(f"[AuditService] JSON解析成功，提取的长度: {len(json_content)}")
            return result

        except (json.JSONDecodeError, IndexError, ValueError) as e:
            # 如果智能提取失败，尝试传统方法：移除markdown代码块
            try:
                response_text = response.strip()

                # 尝试移除 ```json ... ``` 格式
                if "```json" in response_text:
                    parts = response_text.split("```json")
                    if len(parts) > 1:
                        response_text = parts[1].split("```")[0].strip()
                        if response_text:
                            return json.loads(response_text)

                # 尝试移除 ``` ... ``` 格式
                if "```" in response_text:
                    parts = response_text.split("```")
                    # 取第二个```块（如果有的话）
                    if len(parts) >= 3:
                        response_text = parts[1].strip()
                        if response_text:
                            return json.loads(response_text)
                    elif len(parts) == 2:
                        # 只有两个```，取中间的内容
                        response_text = parts[1].strip()
                        # 移除可能的lang标识（第一行）
                        lines = response_text.split('\n', 1)
                        if len(lines) > 1:
                            response_text = lines[1].strip()
                        if response_text:
                            return json.loads(response_text)

                logger.warning(f"[AuditService] JSON解析失败: {e}, 原始响应前200字符: {response[:200]}")
                return None

            except Exception as e2:
                logger.warning(f"[AuditService] JSON解析失败（所有方法）: {e}, 原始响应前200字符: {response[:200]}")
                return None

    def _format_event_content(self, event_content: str) -> str:
        """
        格式化事件内容

        如果event_content是JSON字符串，则美化格式
        否则直接返回原字符串
        """
        try:
            event_content_detail = json.loads(event_content)
            return json.dumps(event_content_detail, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            return event_content


# 全局服务实例（延迟初始化）
_audit_service_instance: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """获取AuditService单例"""
    global _audit_service_instance
    if _audit_service_instance is None:
        _audit_service_instance = AuditService()
    return _audit_service_instance
