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
                return R.ok(message="忽略松开事件", data={"ignored": True, "reason": "state=false"})

            # 保存图片到会话目录
            log_step(1, 4, "保存图片", sessionID)
            filename = pic_filename or f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = get_session_storage_service().save_image(sessionID, filename, image_data)

            # 读取图片并转换为base64
            log_step(2, 4, "准备图片数据", sessionID)
            image_base64 = get_session_storage_service().get_image_base64(image_path)
            if not image_base64:
                logger.error(f"[AuditService] [{sessionID}] ❌ 图片读取失败")
                return R.error(message="图片读取失败", code="500")

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
            return R.error(message="审计处理失败", data=str(e), code="500")

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

        log_step(5, 4, "生成审计结果", sessionID)

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

            return R.error(
                message="发现安全风险",
                code="300001",
                data=result_data.model_dump()
            )
        else:
            # 无风险：返回正常
            logger.info(f"[AuditService] [{sessionID}] ✅ 操作正常，无风险")

            # 如果有流程名称，附加到返回信息
            response_data = {
                "equipment_asset": sessionID,
                "work_content": summary
            }
            if process_name:
                response_data["process_name"] = process_name
                logger.info(f"[AuditService] [{sessionID}] 流程: {process_name}")

            logger.info("=" * 60)

            return R.ok(
                message="操作正常",
                data=response_data
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

        system_prompt = """你是一个专业的运维安全审计AI助手。你的任务是判断运维操作是否存在安全风险。

【风险等级分类】

**高危操作（has_risk=true, risk_level="high"）**：
- 删除关键数据、格式化磁盘、停止核心服务
- 修改生产环境核心配置（防火墙、数据库、网络、系统配置）
- 执行未知来源的脚本或命令
- 危险命令：rm -rf /、dd、shutdown、format、delete from等
- 在生产环境执行非授权的修改操作
- 修改用户权限、创建管理员账户

**中危操作（has_risk=true, risk_level="medium"）**：
- 修改非关键配置文件
- 重启非核心服务
- 可能影响性能的操作
- 安装新软件或包
- 操作流程不规范但未造成直接影响

**低危操作（has_risk=true, risk_level="low"）**：
- 轻微操作不规范
- 操作顺序有误但无安全影响
- 查看敏感信息（如日志、配置）

**安全操作（has_risk=false, risk_level="none"）**：
- 查询类操作、常规查看操作
- 打开程序、浏览文件
- 符合规范的常规操作
- 鼠标点击界面元素（如按钮、菜单）

【判断要点】
- 结合截图内容，判断实际操作的上下文
- event_type 告诉你操作类型（鼠标点击、键盘输入、命令等）
- event_content 提供操作的技术细节
- 同样的操作在不同上下文可能有不同的风险等级

【输出要求】
必须且只能返回纯JSON格式，不要包含任何其他文字、不要使用markdown代码块（```）、不要添加任何解释：
{
  "has_risk": true或false,
  "risk_level": "high/medium/low/none",
  "alarm_message": "具体告警内容（20-100字，说明为什么有风险）"
}

重要提示：
- 直接输出JSON对象本身，不要用```包裹
- alarm_message在有风险时必须详细说明原因
- 如果是安全操作，alarm_message可以为空字符串"""

        user_prompt = f"""请审计以下运维操作：

事件类型：{audit_opt.event_type}
事件详情：{event_content_display}

请结合截图内容和事件类型，判断该操作是否存在安全风险，并严格按JSON格式返回结果。"""

        try:
            response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_base64,
                temperature=0.1,
                max_tokens=500,
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
        """
        try:
            response_text = response.strip()

            # 移除可能的markdown代码块标记
            # 处理 ```json ... ``` 格式
            if "```json" in response_text:
                parts = response_text.split("```json")
                if len(parts) > 1:
                    response_text = parts[1].split("```")[0].strip()
            # 处理 ``` ... ``` 格式（无json标记）
            elif response_text.startswith("```"):
                parts = response_text.split("```")
                if len(parts) > 1:
                    response_text = parts[1].strip()
                    # 如果后面还有```，取中间部分
                    if "```" in response_text:
                        response_text = response_text.split("```")[0].strip()

            # 如果提取后的内容为空，记录详细日志并返回None
            if not response_text:
                logger.warning(f"[AuditService] 提取后的内容为空，原始响应: {response[:200]}")
                return None

            return json.loads(response_text)

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"[AuditService] JSON解析失败: {e}, 原始响应: {response[:200]}")
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
