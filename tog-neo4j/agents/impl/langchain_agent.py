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
from agents.tools.langchain_tools import get_all_tools
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

        # 获取工具
        self.tools = get_all_tools()

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
            max_iterations=5,
            max_execution_time=120,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        logger.info("✅ LangChainQueryAgent 初始化完成")

    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个专业的知识图谱查询助手。你的任务是根据用户问题，选择最合适的查询工具来获取答案。

可用工具说明:
1. **graphrag_query**: 适用于简单的事实查询和信息检索
2. **tog_query**: 适用于需要多步推理和逻辑链的复杂问题
3. **hybrid_query**: 适用于需要深度推理和广泛检索的复杂问题

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
                    
                    logger.info(f"[{kb_grag_id}] 🔍 尝试查询知识库: {kb_name}")
                    
                    # 创建新的上下文，使用当前知识库ID
                    kb_context = AgentContext(
                        grag_id=kb_grag_id,
                        question=context.question,
                        conversation_history=context.conversation_history,
                        metadata=context.metadata
                    )
                    
                    # 准备输入
                    agent_input = self._prepare_input(kb_context)

                    try:
                        # 执行 Agent
                        result = await self.agent_executor.ainvoke(agent_input)
                        
                        # 检查结果是否有效（例如，答案长度大于一定阈值）
                        output = result.get("output", "")
                        if output and len(output.strip()) > 20:  # 简单的有效性检查
                            logger.info(f"[{kb_grag_id}] ✅ 在知识库 '{kb_name}' 中找到有效答案")
                            
                            execution_time = time.time() - start_time
                            return AgentResult(
                                success=True,
                                data={
                                    "question": context.question,
                                    "answer": output,
                                    "grag_id": kb_grag_id,
                                    "kb_name": kb_name,
                                    "tools_used": self._extract_tools_used(result)
                                },
                                message=f"在知识库 '{kb_name}' 中查询成功",
                                execution_time=execution_time,
                                metadata={
                                    "tools_used": self._extract_tools_used(result),
                                    "kb_used": kb_name,
                                    "kb_grag_id": kb_grag_id
                                }
                            )
                        else:
                            logger.info(f"[{kb_grag_id}] ⚠️ 知识库 '{kb_name}' 查询结果无效，继续尝试下一个")
                            continue
                    except Exception as kb_error:
                        logger.error(f"[{kb_grag_id}] ❌ 查询知识库 '{kb_name}' 时出错: {kb_error}")
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
                
                # 准备输入
                agent_input = self._prepare_input(context)

                # 执行 Agent
                result = await self.agent_executor.ainvoke(agent_input)

                # 解析结果
                return self._parse_result(result, context, time.time() - start_time)

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