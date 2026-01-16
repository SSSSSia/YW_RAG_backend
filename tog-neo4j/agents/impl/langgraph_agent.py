"""
基于 LangGraph 的高级 Agent - 支持复杂工作流和回退策略
"""
import time
from typing import Dict, Any, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agents.base import BaseAgent, AgentContext, AgentResult
from agents.tools.langchain_tools import GraphRAGTool, ToGTool, HybridQueryTool
from core.config import settings
from utils.logger import logger
from utils.java_backend import get_knowledge_bases
import operator


# ==================== 状态定义 ====================

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    grag_id: str
    question: str
    all_kbs: list
    attempted_kbs: list
    current_answer: str
    complexity: str
    selected_method: str
    is_success: bool
    error_count: int


# ==================== LangGraph Agent ====================

class LangGraphAgent(BaseAgent):
    """基于 LangGraph 的高级 Agent"""

    def __init__(self):
        super().__init__(
            name="LangGraphAgent",
            description="使用 LangGraph 实现的高级 Agent，支持知识库回退和复杂工作流"
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

        # 初始化工具
        self.graphrag_tool = GraphRAGTool()
        self.tog_tool = ToGTool()
        self.hybrid_tool = HybridQueryTool()

        # 构建工作流
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        logger.info("✅ LangGraphAgent 初始化完成")

    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("analyze_complexity", self._analyze_complexity)
        workflow.add_node("select_method", self._select_method)
        workflow.add_node("query_primary_kb", self._query_primary_kb)
        workflow.add_node("validate_answer", self._validate_answer)
        workflow.add_node("fallback_query", self._fallback_query)
        workflow.add_node("final_answer", self._final_answer)

        # 设置入口
        workflow.set_entry_point("analyze_complexity")

        # 添加边
        workflow.add_edge("analyze_complexity", "select_method")
        workflow.add_edge("select_method", "query_primary_kb")

        # 条件边: 验证答案后的路由
        workflow.add_conditional_edges(
            "query_primary_kb",
            self._should_fallback,
            {
                "validate": "validate_answer",
                "fallback": "fallback_query",
                "end": "final_answer"
            }
        )

        workflow.add_conditional_edges(
            "validate_answer",
            self._after_validation,
            {
                "success": "final_answer",
                "fallback": "fallback_query"
            }
        )

        workflow.add_conditional_edges(
            "fallback_query",
            self._after_fallback,
            {
                "continue": "fallback_query",
                "end": "final_answer"
            }
        )

        workflow.add_edge("final_answer", END)

        return workflow

    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理该任务"""
        return True

    async def execute(self, context: AgentContext) -> AgentResult:
        """执行 Agent"""
        start_time = time.time()

        try:
            logger.info(f"[{context.grag_id}] 🤖 LangGraphAgent 开始执行")

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
                    
                    # 初始化状态，使用当前知识库
                    initial_state = {
                        "messages": [HumanMessage(content=kb_context.question)],
                        "grag_id": kb_grag_id,
                        "question": kb_context.question,
                        "all_kbs": [kb],  # 只查询当前知识库
                        "attempted_kbs": [kb_grag_id],
                        "current_answer": "",
                        "complexity": "",
                        "selected_method": "",
                        "is_success": False,
                        "error_count": 0
                    }

                    try:
                        # 执行工作流
                        final_state = await self.app.ainvoke(initial_state)
                        
                        # 检查结果是否有效（例如，答案长度大于一定阈值且不包含错误信息）
                        current_answer = final_state.get("current_answer", "")
                        if current_answer and len(current_answer.strip()) > 20 and "失败" not in current_answer and "错误" not in current_answer and "未找到" not in current_answer:
                            logger.info(f"[{kb_grag_id}] ✅ 在知识库 '{kb_name}' 中找到有效答案")
                            
                            execution_time = time.time() - start_time
                            return AgentResult(
                                success=final_state["is_success"],
                                data={
                                    "question": context.question,
                                    "answer": final_state["current_answer"],
                                    "grag_id": kb_grag_id,
                                    "kb_name": kb_name,
                                    "attempted_kbs": final_state["attempted_kbs"]
                                },
                                message=f"在知识库 '{kb_name}' 中查询成功",
                                execution_time=execution_time,
                                metadata={
                                    "complexity": final_state["complexity"],
                                    "method": final_state["selected_method"],
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
                
                # 获取所有知识库用于回退策略
                all_knowledge_bases = await get_knowledge_bases()
                all_kb_dict = {kb.get("graph_key") or kb.get("grag_id"): kb for kb in all_knowledge_bases}
                current_kb = all_kb_dict.get(context.grag_id)
                all_kbs_for_fallback = [current_kb] if current_kb else []

                # 初始化状态
                initial_state = {
                    "messages": [HumanMessage(content=context.question)],
                    "grag_id": context.grag_id,
                    "question": context.question,
                    "all_kbs": all_kbs_for_fallback,  # 当前知识库用于回退
                    "attempted_kbs": [],
                    "current_answer": "",
                    "complexity": "",
                    "selected_method": "",
                    "is_success": False,
                    "error_count": 0
                }

                # 执行工作流
                final_state = await self.app.ainvoke(initial_state)

                # 构建结果
                execution_time = time.time() - start_time

                return AgentResult(
                    success=final_state["is_success"],
                    data={
                        "question": context.question,
                        "answer": final_state["current_answer"],
                        "grag_id": final_state.get("grag_id"),
                        "attempted_kbs": final_state["attempted_kbs"]
                    },
                    message="查询成功" if final_state["is_success"] else "查询失败",
                    execution_time=execution_time,
                    metadata={
                        "complexity": final_state["complexity"],
                        "method": final_state["selected_method"],
                        "fallback_used": len(final_state["attempted_kbs"]) > 1
                    }
                )

        except Exception as e:
            logger.error(f"LangGraphAgent 执行失败: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data=None,
                message="查询失败",
                error=str(e),
                execution_time=time.time() - start_time
            )

    # ==================== 节点函数 ====================

    async def _analyze_complexity(self, state: AgentState) -> AgentState:
        """分析问题复杂度"""
        logger.info("📊 分析问题复杂度...")

        prompt = ChatPromptTemplate.from_template(
            "分析以下问题的复杂度，返回: simple, moderate, 或 complex\n问题: {question}"
        )

        chain = prompt | self.planning_llm
        response = await chain.ainvoke({"question": state["question"]})

        complexity = "moderate"
        content = response.content.lower()
        if "simple" in content:
            complexity = "simple"
        elif "complex" in content:
            complexity = "complex"

        state["complexity"] = complexity
        logger.info(f"✅ 复杂度: {complexity}")

        return state

    async def _select_method(self, state: AgentState) -> AgentState:
        """选择查询方法"""
        logger.info("🎯 选择查询方法...")

        complexity = state["complexity"]
        question = state["question"].lower()

        # 检查显式指定
        if "tog" in question or "思维图" in question:
            method = "tog"
        elif "graphrag" in question:
            method = "graphrag"
        elif "混合" in question or "hybrid" in question:
            method = "hybrid"
        else:
            # 根据复杂度选择
            if complexity == "simple":
                method = "graphrag"
            elif complexity == "complex":
                method = "hybrid"
            else:
                method = "tog"

        state["selected_method"] = method
        logger.info(f"✅ 选择方法: {method}")

        return state

    async def _query_primary_kb(self, state: AgentState) -> AgentState:
        """查询主知识库"""
        grag_id = state["grag_id"]
        question = state["question"]
        method = state["selected_method"]

        logger.info(f"🔍 查询主知识库: {grag_id}, 方法: {method}")

        try:
            if method == "graphrag":
                result = self.graphrag_tool._run(grag_id, question)
            elif method == "tog":
                result = self.tog_tool._run(grag_id, question)
            else:
                result = await self.hybrid_tool._arun(grag_id, question)

            state["current_answer"] = result
            state["attempted_kbs"].append(grag_id)

            # 简单判断成功
            if "成功" in result and len(result) > 50:
                state["is_success"] = True

        except Exception as e:
            logger.error(f"查询失败: {e}")
            state["current_answer"] = f"查询失败: {str(e)}"
            state["error_count"] += 1

        return state

    async def _validate_answer(self, state: AgentState) -> AgentState:
        """验证答案质量"""
        logger.info("✅ 验证答案质量...")

        answer = state["current_answer"]

        # 简单验证规则
        if len(answer) > 50 and "失败" not in answer and "错误" not in answer:
            state["is_success"] = True
        else:
            state["is_success"] = False

        return state

    async def _fallback_query(self, state: AgentState) -> AgentState:
        """回退查询其他知识库"""
        all_kbs = state["all_kbs"]
        attempted = state["attempted_kbs"]

        # 找到未尝试的知识库
        remaining_kbs = [kb for kb in all_kbs if kb.get("graph_key") not in attempted]

        if not remaining_kbs:
            logger.warning("⚠️ 没有更多知识库可尝试")
            return state

        # 尝试下一个知识库
        next_kb = remaining_kbs[0]
        next_grag_id = next_kb.get("graph_key")

        logger.info(f"🔄 回退查询知识库: {next_kb.get('name')}")

        question = state["question"]
        method = state["selected_method"]

        try:
            if method == "graphrag":
                result = self.graphrag_tool._run(next_grag_id, question)
            elif method == "tog":
                result = self.tog_tool._run(next_grag_id, question)
            else:
                result = await self.hybrid_tool._arun(next_grag_id, question)

            state["current_answer"] = result
            state["attempted_kbs"].append(next_grag_id)
            state["grag_id"] = next_grag_id

            if "成功" in result and len(result) > 50:
                state["is_success"] = True

        except Exception as e:
            logger.error(f"回退查询失败: {e}")
            state["error_count"] += 1

        return state

    async def _final_answer(self, state: AgentState) -> AgentState:
        """生成最终答案"""
        logger.info("📝 生成最终答案")

        if not state["is_success"]:
            state["current_answer"] = f"很抱歉，在 {len(state['attempted_kbs'])} 个知识库中都未找到满意的答案。"

        return state

    # ==================== 条件判断函数 ====================

    def _should_fallback(self, state: AgentState) -> str:
        """判断是否需要回退"""
        if state["is_success"]:
            return "validate"
        elif state["error_count"] > 3:
            return "end"
        elif len(state["all_kbs"]) > 1:
            return "fallback"
        else:
            return "end"

    def _after_validation(self, state: AgentState) -> str:
        """验证后的路由"""
        if state["is_success"]:
            return "success"
        else:
            return "fallback"

    def _after_fallback(self, state: AgentState) -> str:
        """回退后的路由"""
        if state["is_success"]:
            return "end"
        elif len(state["attempted_kbs"]) >= len(state["all_kbs"]):
            return "end"
        elif state["error_count"] > 3:
            return "end"
        else:
            return "continue"