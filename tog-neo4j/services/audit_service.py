"""
AI审计服务（优化版）- 完全基于LLM的风险审计，使用MySQL存储操作记录
性能优化：
1. 合并风险审计和操作总结为一次LLM调用
2. 内存中直接转换图片，减少I/O操作
"""
import os
import json
import base64
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

            # 【优化】在内存中直接转换图片为base64，避免重复I/O
            log_step(1, 3, "准备图片数据", sessionID)
            filename = pic_filename or f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

            # 判断图片MIME类型（根据文件扩展名）
            ext = os.path.splitext(filename)[1].lower() if '.' in filename else '.jpg'
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp"
            }.get(ext, "image/jpeg")

            # 转换为base64并添加data URL前缀
            base64_str = base64.b64encode(image_data).decode('utf-8')
            image_base64 = f"data:{mime_type};base64,{base64_str}"

            # 保存图片到会话目录
            log_step(2, 3, "保存图片", sessionID)
            image_path = get_session_storage_service().save_image(sessionID, filename, image_data)

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
        纯LLM风险审计（核心逻辑 - 优化版）

        【性能优化】合并两次LLM调用为一次：
        1. 同时获取风险判断和操作总结
        2. 减少视觉模型调用次数（从2次→1次）
        3. 保存操作记录到MySQL数据库
        """
        log_step(3, 3, "LLM智能审计（风险+总结）", sessionID)

        # 【优化】一次LLM调用同时完成风险审计和总结生成
        audit_result, summary = await self._audit_and_summary(sessionID, audit_opt, image_base64)

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
                code="30001",
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

    async def _audit_and_summary(
        self,
        sessionID: str,
        audit_opt: AuditOpt,
        image_base64: str
    ) -> tuple:
        """
        【优化】一次LLM调用同时完成风险审计和操作总结

        返回: (audit_result, summary)
        - audit_result: {"has_risk": bool, "risk_level": str, "alarm_message": str}
        - summary: str (操作总结)
        """
        event_content_display = self._format_event_content(audit_opt.event_content)

        # 【优化】获取最近的操作历史作为上下文（最近10条，减少token消耗）
        recent_records = get_operation_db().get_records_by_session(sessionID)
        recent_operations = []
        # 额外提取历史图片的summary（用户需求）
        previous_image_summaries = []
        for record in recent_records[-10:]:  # 从20条减少到10条
            try:
                op_data = json.loads(record['operation'])
                recent_operations.append({
                    'event_type': op_data.get('event_type', 'unknown'),
                    'event_time': op_data.get('event_time', ''),
                    'summary': record.get('summary', '')
                })
                # 收集历史图片的summary（如果有）
                if record.get('summary') and record.get('image_path'):
                    previous_image_summaries.append({
                        'summary': record.get('summary'),
                        'event_type': op_data.get('event_type', 'unknown')
                    })
            except:
                pass

        # 构建历史上下文
        history_context = ""
        if recent_operations:
            history_context = "\n\n【最近操作历史】\n"
            for i, op in enumerate(recent_operations[-5:], 1):  # 只显示最近5条
                history_context += f"{i}. {op['event_type']} - {op['summary']}\n"

        # 【新增】构建历史图片总结参考（用户需求）
        image_summaries_reference = ""
        if previous_image_summaries:
            image_summaries_reference = "\n\n【历史图片总结参考】\n"
            image_summaries_reference += "以下是该会话之前上传图片的AI审查总结，请作为参考来理解当前操作的上下文：\n"
            for i, img_sum in enumerate(previous_image_summaries[-5:], 1):  # 最多显示最近5张图片的总结
                image_summaries_reference += f"{i}. 操作类型: {img_sum['event_type']}, 总结: {img_sum['summary']}\n"
            image_summaries_reference += "\n⚠️ 注意：这些历史图片总结可以帮助你更好地理解当前操作的上下文和连续性。\n"

        # 优化后的系统提示词（合并版）- 参考summary接口的详细流程白名单
        system_prompt = """你是运维安全审计AI，需要同时完成风险判断和操作总结。

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

🔍 **【重要】历史上下文关联判断规则：**
必须根据【最近操作历史】来判断当前操作的合理性：
1. 如果上一步是"点击Root账户"，紧接着的键盘输入（任意按键）都属于"输入Root密码"流程的一部分，判定为安全
2. 如果上一步是"点击确认(C)"，紧接着的键盘输入属于"再次输入Root密码"，判定为安全
3. 如果上一步是"点击安装目的地"，紧接着的"点击完成(D)"属于流程的一部分，判定为安全
4. 如果当前操作看起来独立，但结合历史上下文后能识别出是白名单流程的继续，则判定为安全
5. **【特别规则】如果当前操作是白名单流程的第1步（如"Test this media & install"或点击Kylin菜单），则视为新的流程开始，判定为安全，即使历史中已有完整流程**
6. 只有在历史上下文和当前操作都无法匹配白名单流程时，才标记为异常

【风险等级判断】
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
- 严格匹配上述两个白名单流程的操作
- 点击应用按钮、菜单（在流程内）
- 应用内部操作（不涉及文件访问）

【关键规则】
- 必须识别截图中的文件路径！
- 打开文件夹/文件 = 有风险（除非在上述白名单流程中）
- 如果当前操作不在上述两个流程白名单中，必须报告风险

【输出格式】纯JSON（不要```）：
{
    "has_risk": true/false,
    "risk_level": "high/medium/low/none",
    "alarm_message": "风险原因（20-100字，无风险时填空字符串）",
    "summary": "操作总结（一句话30字以内，说明操作类型和目标）"
}"""

        user_prompt = f"""请审计以下运维操作并生成总结：
{history_context}
{image_summaries_reference}

【当前待审计的操作】
事件类型：{audit_opt.event_type}
事件详情：{event_content_display}

⚠️ **必须结合【最近操作历史】和【历史图片总结参考】来判断当前操作：**
1. 如果上一步是"点击Root账户"或包含"Root账户"、"密码设置"等关键词，紧接着的键盘输入（任意按键）都属于"输入Root密码"流程的一部分 → 判定为安全
2. 如果上一步是点击某个输入框或确认按钮，紧接着的键盘输入属于密码输入 → 判定为安全
3. 如果当前操作独立看像是异常，但结合历史上下文后能识别出是白名单流程的继续 → 判定为安全
4. **【关键】如果当前操作是白名单流程的第1步（如点击"Test this media & install"或点击Kylin启动菜单），则视为新的流程开始 → 判定为安全，即使历史中已有完整流程**
5. **【新增】参考【历史图片总结参考】中的信息，理解当前操作在整个会话中的上下文和连续性，避免误判**

请结合截图内容、事件类型、最近的操作历史以及历史图片总结参考：
1. 判断该操作是否在上述白名单流程中（必须结合历史上下文）
2. 判断该操作是否存在安全风险
3. 生成简洁的操作总结（一句话30字以内）

⚠️ 关键判断原则：
- **优先参考历史上下文**：如果上一步操作能解释当前操作（如"点击Root账户"后的按键输入），则判定为安全
- **利用历史图片总结**：历史图片的AI审查总结可以帮助你理解操作的连续性和上下文，例如如果历史显示正在进行系统安装流程，当前操作应视为流程的一部分
- **识别新流程开始**：如果当前操作是白名单流程的第1步，视为新的流程开始，判定为安全（不要误判为"重复安装"）
- 如果当前操作严格匹配上述17步系统重装流程或8步密码重置流程中的某一步 → 判定为安全
- 只有在历史上下文和历史图片总结参考都显示无法匹配白名单流程时，才报告风险

严格按照JSON格式返回所有字段。"""

        try:
            # 【优化】一次视觉模型调用完成所有任务
            response = llm_client.chat_with_vision(
                prompt=user_prompt,
                image_base64=image_base64,
                temperature=0.3,
                max_tokens=2000,  # 增加token以同时输出风险和总结
                system_prompt=system_prompt
            )

            if not response:
                logger.warning(f"[AuditService] [{sessionID}] LLM未返回响应，使用默认值")
                return self._get_default_result(audit_opt)

            result = self._parse_json_response(response)
            if not result:
                logger.warning(f"[AuditService] [{sessionID}] JSON解析失败，使用默认值")
                return self._get_default_result(audit_opt)

            audit_result = {
                "has_risk": result.get("has_risk", False),
                "risk_level": result.get("risk_level", "none"),
                "alarm_message": result.get("alarm_message", "")
            }
            summary = result.get("summary", self._get_default_summary(audit_opt))

            return audit_result, summary

        except Exception as e:
            logger.error(f"[AuditService] [{sessionID}] 审计失败: {e}")
            return self._get_default_result(audit_opt)

    def _get_default_result(self, audit_opt: AuditOpt) -> tuple:
        """返回默认的审计结果（降级策略）"""
        # 关键词匹配作为降级策略
        error_keywords = [
            "删除", "delete", "drop", "truncate",
            "格式化", "format", "rm -rf",
            "shutdown", "停止", "stop"
        ]
        event_text = audit_opt.event_content.lower()
        has_risk = any(keyword.lower() in event_text for keyword in error_keywords)

        if has_risk:
            audit_result = {
                "has_risk": True,
                "risk_level": "medium",
                "alarm_message": "检测到可能的危险操作关键词"
            }
        else:
            audit_result = {
                "has_risk": False,
                "risk_level": "none",
                "alarm_message": ""
            }

        summary = self._get_default_summary(audit_opt)
        return audit_result, summary

    def _get_default_summary(self, audit_opt: AuditOpt) -> str:
        """生成默认的操作总结（无LLM调用）"""
        if audit_opt.event_type == "ws_mouse_click":
            return "鼠标点击操作"
        elif audit_opt.event_type == "ws_keyboard":
            return "键盘输入操作"
        elif "command" in audit_opt.event_type.lower():
            return "命令执行操作"
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
