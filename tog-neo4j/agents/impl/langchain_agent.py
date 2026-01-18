"""
基于 LangChain 的智能查询 Agent
"""
import time
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from agents.base import BaseAgent, AgentContext, AgentResult
from agents.tools.langchain_tools import get_all_tools, GraphRAGTool, ToGTool, HybridQueryTool
from core.config import settings
from utils.logger import logger
from utils.java_backend import get_knowledge_bases


class LangChainQueryAgent(BaseAgent):
    """基于 LangChain 的智能查询 Agent"""

    def __init__(self):
        super().__init__(
            name="LangChainQueryAgent",
            description="使用 LangChain 框架的智能查询 Agent，支持自动工具选择和推理"
        )

        # 初始化 Ollama LLM
        self.llm = ChatOllama(
            model=settings.llm_model,
            temperature=0.7,
            base_url=settings.llm_api_url,
            timeout=120
        )

        # 初始化规划用的 Ollama LLM (低温度)
        self.planning_llm = ChatOllama(
            model=settings.llm_model,
            temperature=0.1,
            base_url=settings.llm_api_url,
            timeout=120
        )

        # 获取工具
        self.tools = get_all_tools()

        # 初始化工具实例
        self.graphrag_tool = GraphRAGTool()
        self.tog_tool = ToGTool()
        self.hybrid_tool = HybridQueryTool()

        # 创建 Agent Prompt - 定义与LLM交互的消息结构
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),  # 系统角色消息，定义Agent的行为和规则
            MessagesPlaceholder(variable_name="chat_history", optional=True),  # 对话历史占位符，可选，用于保持多轮对话上下文
            ("human", "{input}"),                   # 人类用户输入占位符，接收用户的具体问题
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent工作区占位符，用于存储工具调用和思考过程
        ])

        # 创建 Agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # 创建 Agent Executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,  # 增加最大迭代次数以处理更复杂的查询
            max_execution_time=300,  # 增加最大执行时间为5分钟
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        logger.info("✅ LangChainQueryAgent 初始化完成")

    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个专业的知识图谱查询助手。你的任务是根据用户问题，选择最合适的查询工具来获取答案。
        
        可用工具说明:
        1. **graphrag_query**: 适用于简单的事实查询和信息检索，检索速度较快
        2. **tog_query**: 适用于需要多步推理和逻辑链的复杂问题，检索速度较慢
        3. **hybrid_query**: 适用于需要深度推理和广泛检索的复杂问题，检索速度最慢
        
        工具选择策略:
        - 简单问题 (如"什么是X"、"X的定义") → 使用 graphrag_query
        - 中等复杂问题 (如"X和Y的关系"、"X如何影响Y") → 使用 tog_query
        - 复杂问题 (如"分析X的多方面影响"、"比较X和Y的优缺点") → 使用 hybrid_query
        
        重要提示:
        1. 首先分析问题复杂度
        2. 选择最合适的工具
        3. 如果第一次查询失败或结果不满意，可以尝试其他工具
        4. 提供清晰、结构化的答案
        5. 如果查询失败，要提供有用的错误信息
        
        请始终以用户的问题为中心，提供准确、有用的答案。"""

    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理该任务"""
        # LangChain Agent 可以处理所有查询任务
        return True

    async def execute(self, context: AgentContext) -> AgentResult:
        """执行查询"""
        start_time = time.time()

        try:
            logger.info(f"[{context.grag_id}] 🤖 LangChainQueryAgent 开始执行")
            logger.info(f"[{context.grag_id}] 💬 问题: {context.question}")

            # 如果 grag_id 为空或为 default，则获取所有知识库并循环查询
            if not context.grag_id or context.grag_id == "default":
                logger.info("📚 grag_id 为空，开始获取知识库列表并循环查询")

                # 获取知识库列表
                knowledge_bases = await get_knowledge_bases()

                if not knowledge_bases:
                    logger.warning("❌ 未找到任何知识库")
                    return AgentResult(
                        success=False,
                        data=None,
                        message="未找到任何可用的知识库",
                        error="no_knowledge_bases_found",
                        execution_time=time.time() - start_time
                    )

                logger.info(f"✅ 获取到 {len(knowledge_bases)} 个知识库: {[kb['name'] for kb in knowledge_bases]}")

                # 遍历知识库进行查询
                for kb in knowledge_bases:
                    kb_grag_id = kb.get("graph_key") or kb.get("grag_id")
                    kb_name = kb.get("name", "未知知识库")

                    # 检查是否获取到了有效的 grag_id
                    if not kb_grag_id:
                        logger.warning(f"⚠️ 知识库 '{kb_name}' 缺少 graph_key 或 grag_id，跳过")
                        continue

                    logger.info(f"[{kb_grag_id}] 🔍 尝试查询知识库: {kb_name}")

                    # 创建新的上下文，使用当前知识库ID
                    kb_context = AgentContext(
                        grag_id=kb_grag_id,
                        question=context.question,
                        conversation_history=context.conversation_history,
                        metadata=context.metadata
                    )

                    # 执行查询（不包含验证逻辑）
                    result = await self._execute_single_query(kb_context, kb_name, validate_answer=False)

                    # 在外层循环中进行大模型验证，避免每个知识库都验证
                    if result.data and result.data.get("answer"):
                        answer = result.data["answer"]
                        is_valid = await self._validate_answer(context.question, answer)

                        if is_valid:
                            logger.info(f"[{kb_grag_id}] ✅ 在知识库 '{kb_name}' 中找到有效答案")

                            execution_time = time.time() - start_time
                            result.execution_time = execution_time
                            result.success = True
                            result.message = f"在知识库 '{kb_name}' 中查询成功"
                            return result
                        else:
                            logger.info(f"[{kb_grag_id}] ⚠️ 知识库 '{kb_name}' 答案验证失败，继续尝试下一个")
                            continue
                    else:
                        logger.info(f"[{kb_grag_id}] ⚠️ 知识库 '{kb_name}' 查询失败，继续尝试下一个")
                        continue

                # 所有知识库都查询失败
                logger.error("❌ 所有知识库查询都失败")
                execution_time = time.time() - start_time
                return AgentResult(
                    success=False,
                    data=None,
                    message="在所有知识库中都未能找到满意答案",
                    error="all_knowledge_bases_failed",
                    execution_time=execution_time
                )
            else:
                # 原有逻辑：指定知识库查询
                logger.info(f"[{context.grag_id}] 🎯 执行指定知识库查询")

                # 执行完整的查询流程
                result = await self._execute_single_query(context, "指定知识库")
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                return result

        except Exception as e:
            logger.error(f"LangChainQueryAgent 执行失败: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data=None,
                message="查询失败",
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _prepare_input(self, context: AgentContext) -> Dict[str, Any]:
        """准备 Agent 输入"""
        # 转换对话历史为 LangChain 格式
        chat_history = []
        for msg in context.conversation_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # 构建输入
        input_text = f"""
知识库ID: {context.grag_id}
用户问题: {context.question}

请选择合适的工具查询并返回答案。
"""

        return {
            "input": input_text,
            "chat_history": chat_history
        }

    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
        """从结果中提取使用的工具"""
        intermediate_steps = result.get("intermediate_steps", [])
        tools_used = []
        for step in intermediate_steps:
            if len(step) > 0:
                action = step[0]
                tools_used.append(action.tool)
        return tools_used

    def _parse_result(
        self,
        result: Dict[str, Any],
        context: AgentContext,
        execution_time: float
    ) -> AgentResult:
        """解析 Agent 执行结果"""
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # 提取使用的工具
        tools_used = self._extract_tools_used(result)

        logger.info(f"✅ 查询完成，使用的工具: {', '.join(tools_used)}")
        logger.info(f"⏱️ 总耗时: {execution_time:.2f}秒")

        return AgentResult(
            success=True,
            data={
                "question": context.question,
                "answer": output,
                "grag_id": context.grag_id,
                "tools_used": tools_used
            },
            message="查询成功",
            execution_time=execution_time,
            metadata={
                "tools_used": tools_used,
                "intermediate_steps": len(intermediate_steps)
            }
        )

    async def _execute_single_query(
        self,
        context: AgentContext,
        kb_name: str = "未知知识库",
        validate_answer: bool = True
    ) -> AgentResult:
        """执行单个知识库的完整查询流程

        Args:
            context: 查询上下文
            kb_name: 知识库名称
            validate_answer: 是否验证答案质量（默认True，设为False时跳过大模型验证）

        Returns:
            AgentResult: 查询结果
        """
        start_time = time.time()

        # 分析问题复杂度
        complexity = await self._analyze_complexity(context.question)

        # 选择查询方法
        method = await self._select_method(complexity, context.question)

        # 根据选择的方法执行查询
        result = await self._execute_query_with_method(context.grag_id, context.question, method)

        # 记录使用的工具
        tools_used = [method]

        execution_time = time.time() - start_time

        # 根据 validate_answer 参数决定是否验证答案
        is_valid = True
        if validate_answer:
            # 使用大模型验证答案质量
            is_valid = await self._validate_answer(context.question, result)

        return AgentResult(
            success=is_valid,
            data={
                "question": context.question,
                "answer": result,
                "grag_id": context.grag_id,
                "kb_name": kb_name,
                "tools_used": tools_used
            },
            message="查询成功" if is_valid else "查询失败",
            execution_time=execution_time,
            metadata={
                "complexity": complexity,
                "method": method,
                "tools_used": tools_used,
                "kb_used": kb_name,
                "kb_grag_id": context.grag_id
            }
        )

    async def _analyze_complexity(self, question: str) -> str:
        """分析问题复杂度"""
        from langchain_core.prompts import ChatPromptTemplate

        logger.info("📊 分析问题复杂度...")

        prompt = ChatPromptTemplate.from_template(
            "分析以下问题的复杂度，返回: simple, moderate, 或 complex\n问题: {question}"
        )

        chain = prompt | self.planning_llm
        response = await chain.ainvoke({"question": question})

        complexity = "moderate"
        content = response.content.lower()
        if "simple" in content:
            complexity = "simple"
        elif "complex" in content:
            complexity = "complex"

        logger.info(f"✅ 复杂度: {complexity}")

        return complexity

    async def _select_method(self, complexity: str, question: str) -> str:
        """选择查询方法"""
        logger.info("🎯 选择查询方法...")

        question_lower = question.lower()

        # 检查显式指定
        if "tog" in question_lower or "思维图" in question_lower:
            method = "tog"
        elif "graphrag" in question_lower:
            method = "graphrag"
        elif "混合" in question_lower or "hybrid" in question_lower:
            method = "hybrid"
        else:
            # 根据复杂度选择
            if complexity == "simple":
                method = "graphrag"
            elif complexity == "complex":
                method = "hybrid"
            else:
                method = "tog"

        logger.info(f"✅ 选择方法: {method}")

        return method

    async def _execute_query_with_method(self, grag_id: str, question: str, method: str) -> str:
        """根据选择的方法执行查询"""
        logger.info(f"🔍 使用方法 {method} 查询知识库: {grag_id}")

        try:
            if method == "graphrag":
                result = self.graphrag_tool._run(grag_id, question)
            elif method == "tog":
                result = self.tog_tool._run(grag_id, question)
            else:  # hybrid
                result = await self.hybrid_tool._arun(grag_id, question)

            logger.info(f"✅ {method} 查询完成")
            return result

        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _validate_answer(self, question: str, answer: str) -> bool:
        """使用大模型验证答案质量

        Args:
            question: 原始问题
            answer: 待验证的答案

        Returns:
            bool: 答案是否有效
        """
        logger.info("🤖 使用大模型验证答案质量...")

        # 检查答案基本属性
        if not answer or not isinstance(answer, str):
            logger.info("❌ 答案为空或非字符串")
            return False

        # 清理答案文本
        cleaned_answer = answer.strip()

        # # 检查是否明显是错误信息
        # error_indicators = [
        #     "未找到", "not found", "no result",
        #     "error", "exception", "failed", "not available",
        #     "无法回答", "没有找到", "不存在", "暂无", "没有相关信息", "没有可用结果",
        #     "没有检索到相关内容", "检索失败", "查询失败",
        #     "Agent stopped due to max iterations", "max iterations"
        # ]
        #
        # for indicator in error_indicators:
        #     if indicator in cleaned_answer:
        #         logger.info(f"❌ 答案包含错误指示符: {indicator}")
        #         return False

        # 如果答案长度太短，直接视为无效
        if len(cleaned_answer) < 20:
            logger.info(f"❌ 答案长度不足 ({len(cleaned_answer)} 字符)")
            return False

        # 使用大模型评估答案质量
        try:
            from langchain_core.prompts import ChatPromptTemplate

            validation_prompt = ChatPromptTemplate.from_template(
                """请评估以下答案是否有效回答了用户的问题。
                
用户问题: {question}

AI回答: {answer}

请仅回答 "VALID" 如果回答有效，否则回答 "INVALID"。判断标准：
1. 回答是否直接针对问题
2. 回答是否提供了有用信息
3. 回答是否完整或至少提供了部分有用信息
4. 回答是否不是拒绝回答或错误信息

评估:"""
            )

            chain = validation_prompt | self.planning_llm
            response = await chain.ainvoke({"question": question, "answer": answer})

            # 解析大模型的评估结果
            content = response.content.strip().upper()
            # 修复逻辑：必须包含 VALID 且不包含 INVALID，防止 "INVALID" 被错误判定为 True
            is_valid = "VALID" in content and "INVALID" not in content

            logger.info(f"✅ 大模型评估结果: {'有效' if is_valid else '无效'}")
            return is_valid

        except Exception as e:
            logger.error(f"大模型验证答案时出错: {e}")
            # 如果大模型验证失败，使用备用验证方法
            return self._fallback_validate_answer(answer)

    def _fallback_validate_answer(self, answer: str) -> bool:
        """备用答案验证方法"""
        logger.info("🔄 使用备用验证方法")

        # 检查答案基本属性
        if not answer or not isinstance(answer, str):
            logger.info("❌ 答案为空或非字符串")
            return False

        # 清理答案文本
        cleaned_answer = answer.strip()

        # 检查长度
        if len(cleaned_answer) < 20:
            logger.info(f"❌ 答案长度不足 ({len(cleaned_answer)} 字符)")
            return False

        # # 检查是否包含明显的错误信息
        # error_indicators = [
        #     "失败", "错误", "未找到", "not found", "no result",
        #     "error", "exception", "failed", "not available", "暂时无法", "抱歉",
        #     "无法回答", "没有找到", "不存在", "暂无", "没有相关信息", "没有可用结果",
        #     "没有检索到相关内容", "检索失败", "查询失败",
        #     "Agent stopped due to max iterations", "max iterations"
        # ]
        #
        # for indicator in error_indicators:
        #     if indicator in cleaned_answer:
        #         logger.info(f"❌ 答案包含错误指示符: {indicator}")
        #         return False

        logger.info("✅ 通过备用验证")
        return True
