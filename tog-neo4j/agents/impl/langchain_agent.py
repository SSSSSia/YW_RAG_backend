"""
基于 LangChain 的智能查询 Agent - 添加 Markdown 格式化支持
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
            description="使用 LangChain 框架的智能查询 Agent,支持自动工具选择和推理"
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

        # 创建 Agent Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
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
            max_iterations=20,
            max_execution_time=300,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        logger.info("✅ LangChainQueryAgent 初始化完成")

    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个专业的知识图谱查询助手。你的任务是根据用户问题,选择最合适的查询工具来获取答案。

        可用工具说明:
        1. **graphrag_query**: 适用于简单的事实查询和信息检索,检索速度较快
        2. **tog_query**: 适用于需要多步推理和逻辑链的复杂问题,检索速度较慢
        3. **hybrid_query**: 适用于需要深度推理和广泛检索的复杂问题,检索速度最慢

        工具选择策略:
        - 简单问题 (如"什么是X"、"X的定义") → 使用 graphrag_query
        - 中等复杂问题 (如"X和Y的关系"、"X如何影响Y") → 使用 tog_query
        - 复杂问题 (如"分析X的多方面影响"、"比较X和Y的优缺点") → 使用 hybrid_query

        **答案格式要求** (重要):
        1. 使用 Markdown 格式输出答案
        2. 使用清晰的段落分隔,每个段落用空行分开
        3. 使用列表、加粗等 Markdown 元素提高可读性
        4. 多点内容使用编号列表 (1. 2. 3. ...) 或无序列表 (-)
        5. 关键信息使用加粗强调
        6. 保持简洁、结构化的回答风格

        **答案结构示例**:
        ```markdown
        根据查询结果,这里是答案:

        1. **第一点**: 详细说明
        2. **第二点**: 详细说明
        3. **第三点**: 详细说明

        **总结**: 整体总结
        ```

        重要提示:
        1. 首先分析问题复杂度
        2. 选择最合适的工具
        3. 如果第一次查询失败或结果不满意,可以尝试其他工具
        4. 严格按照上述 Markdown 格式要求输出答案
        5. 如果查询失败,要提供有用的错误信息

        请始终以用户的问题为中心,提供准确、有用、格式良好的答案。"""

    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理该任务"""
        return True

    async def execute(self, context: AgentContext) -> AgentResult:
        """执行查询"""
        start_time = time.time()

        try:
            logger.info(f"[{context.grag_id}] 🤖 LangChainQueryAgent 开始执行")
            logger.info(f"[{context.grag_id}] 💬 问题: {context.question}")

            # 如果 grag_id 为空或为 default,则获取所有知识库并循环查询
            if not context.grag_id or context.grag_id == "default":
                logger.info("📚 grag_id 为空,开始获取知识库列表并循环查询")

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

                    if not kb_grag_id:
                        logger.warning(f"⚠️ 知识库 '{kb_name}' 缺少 graph_key 或 grag_id,跳过")
                        continue

                    logger.info(f"[{kb_grag_id}] 🔍 尝试查询知识库: {kb_name}")

                    kb_context = AgentContext(
                        grag_id=kb_grag_id,
                        question=context.question,
                        conversation_history=context.conversation_history,
                        metadata=context.metadata
                    )

                    result = await self._execute_single_query(kb_context, kb_name, validate_answer=False)

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
                            logger.info(f"[{kb_grag_id}] ⚠️ 知识库 '{kb_name}' 答案验证失败,继续尝试下一个")
                            continue
                    else:
                        logger.info(f"[{kb_grag_id}] ⚠️ 知识库 '{kb_name}' 查询失败,继续尝试下一个")
                        continue

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
                logger.info(f"[{context.grag_id}] 🎯 执行指定知识库查询")

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
        chat_history = []
        for msg in context.conversation_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

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

        tools_used = self._extract_tools_used(result)

        logger.info(f"✅ 查询完成,使用的工具: {', '.join(tools_used)}")
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
            validate_answer: 是否验证答案质量

        Returns:
            AgentResult: 查询结果
        """
        start_time = time.time()

        # 分析问题复杂度
        complexity = await self._analyze_complexity(context.question)

        # 选择查询方法
        method = await self._select_method(complexity, context.question)

        # 根据选择的方法执行查询
        raw_result = await self._execute_query_with_method(context.grag_id, context.question, method)

        # 🔥 新增: 格式化为 Markdown
        # formatted_result = await self._format_to_markdown(context.question, raw_result)
        formatted_result = raw_result

        tools_used = [method]

        execution_time = time.time() - start_time

        is_valid = True
        if validate_answer:
            is_valid = await self._validate_answer(context.question, formatted_result)

        return AgentResult(
            success=is_valid,
            data={
                "question": context.question,
                "answer": formatted_result,  # 返回格式化后的结果
                "grag_id": context.grag_id,
                "execution_time": execution_time,
                "kb_name": kb_name
            },
            message="查询成功" if is_valid else "查询失败",
            execution_time=execution_time,
            metadata={
                "complexity": complexity,
                "method": method,
                "tools_used": tools_used
            }
        )

    async def _format_to_markdown(self, question: str, raw_answer: str) -> str:
        """将原始答案格式化为 Markdown 格式

        Args:
            question: 原始问题
            raw_answer: 原始答案

        Returns:
            str: Markdown 格式的答案
        """
        logger.info("📝 格式化答案为 Markdown...")

        try:
            from langchain_core.messages import HumanMessage

            format_prompt = f"""请将以下答案重新格式化为清晰的 Markdown 格式。

原始问题: {question}

原始答案:
{raw_answer}

格式化要求:
1. 使用清晰的段落分隔,每个段落用空行分开
2. 使用 **粗体** 强调关键信息
3. 使用编号列表 (1. 2. 3.) 或无序列表 (-) 组织多点内容
4. 如果有总结,使用 **总结:** 标记
5. 保持内容完整,不要删减信息
6. 确保逻辑清晰,结构分明

请直接输出格式化后的 Markdown 文本,不要添加任何额外说明。"""

            # 使用 invoke 方法
            response = self.llm.invoke([HumanMessage(content=format_prompt)])

            formatted_answer = response.content.strip()
            logger.info("✅ Markdown 格式化完成")
            return formatted_answer

        except Exception as e:
            logger.error(f"Markdown 格式化失败: {e}")
            # 如果格式化失败,返回原始答案
            return raw_answer

    async def _analyze_complexity(self, question: str) -> str:
        """分析问题复杂度"""
        logger.info("📊 分析问题复杂度...")

        try:
            prompt = f"分析以下问题的复杂度,返回: simple, moderate, 或 complex\n问题: {question}"

            # 使用同步调用
            response = self.planning_llm.invoke(prompt)

            complexity = "moderate"
            content = response.content.lower()
            if "simple" in content:
                complexity = "simple"
            elif "complex" in content:
                complexity = "complex"

            logger.info(f"✅ 复杂度: {complexity}")
            return complexity

        except Exception as e:
            logger.error(f"分析复杂度失败: {e}")
            return "moderate"  # 默认返回中等复杂度

    async def _select_method(self, complexity: str, question: str) -> str:
        """选择查询方法"""
        logger.info("🎯 选择查询方法...")

        question_lower = question.lower()

        if "tog" in question_lower or "思维图" in question_lower:
            method = "tog"
        elif "graphrag" in question_lower:
            method = "graphrag"
        elif "混合" in question_lower or "hybrid" in question_lower:
            method = "hybrid"
        else:
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
        """使用大模型验证答案质量"""
        logger.info("🤖 使用大模型验证答案质量...")

        if not answer or not isinstance(answer, str):
            logger.info("❌ 答案为空或非字符串")
            return False

        cleaned_answer = answer.strip()

        if len(cleaned_answer) < 20:
            logger.info(f"❌ 答案长度不足 ({len(cleaned_answer)} 字符)")
            return False

        try:
            from langchain_core.messages import HumanMessage

            validation_prompt = f"""请评估以下答案是否有效回答了用户的问题。
                
用户问题: {question}

AI回答: {answer}

请仅回答 "VALID" 如果回答有效,否则回答 "INVALID"。判断标准:
1. 回答是否直接针对问题
2. 回答是否提供了有用信息
3. 回答是否完整或至少提供了部分有用信息
4. 回答是否不是拒绝回答或错误信息

评估:"""

            # 使用 invoke 方法
            response = self.planning_llm.invoke([HumanMessage(content=validation_prompt)])

            content = response.content.strip().upper()
            is_valid = "VALID" in content and "INVALID" not in content

            logger.info(f"✅ 大模型评估结果: {'有效' if is_valid else '无效'}")
            return is_valid

        except Exception as e:
            logger.error(f"大模型验证答案时出错: {e}")
            return self._fallback_validate_answer(answer)

    def _fallback_validate_answer(self, answer: str) -> bool:
        """备用答案验证方法"""
        logger.info("🔄 使用备用验证方法")

        if not answer or not isinstance(answer, str):
            logger.info("❌ 答案为空或非字符串")
            return False

        cleaned_answer = answer.strip()

        if len(cleaned_answer) < 20:
            logger.info(f"❌ 答案长度不足 ({len(cleaned_answer)} 字符)")
            return False

        logger.info("✅ 通过备用验证")
        return True