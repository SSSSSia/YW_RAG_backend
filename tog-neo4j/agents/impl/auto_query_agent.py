"""
自动查询 Agent - 根据问题自动选择最佳查询方法
支持指定知识库优先查询，失败后自动回退到其他知识库
"""
import time
from agents.base import BaseAgent, AgentContext, AgentResult
from agents.tools import QueryToGTool, QueryGraphRAGTool, QueryHybridTool
from core.llm_client import llm_client
from utils.logger import logger


class AutoQueryAgent(BaseAgent):
    """自动查询 Agent"""

    def __init__(self):
        super().__init__(
            name="AutoQueryAgent",
            description="专门处理用户查询，根据问题复杂度自动选择最佳查询方法，支持跨知识库回退查询"
        )
        # 注册工具
        self.tog_tool = QueryToGTool()
        self.graphrag_tool = QueryGraphRAGTool()
        self.hybrid_tool = QueryHybridTool()

        self.register_tool(self.tog_tool)
        self.register_tool(self.graphrag_tool)
        self.register_tool(self.hybrid_tool)

    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理该任务"""
        query_keywords = ['查询', '搜索', '查找', '告诉我', '什么是', '谁是', '如何', '为什么', 'query', 'search']
        return any(kw in context.question.lower() for kw in query_keywords)

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        执行自动查询（含回退逻辑）
        逻辑：
        1. 如果用户指定了 grag_id，优先尝试查询该知识库。
        2. 如果指定知识库查询成功且答案有效，直接返回。
        3. 如果失败或无有效答案，尝试查询其他可用知识库。
        """
        start_time = time.time()
        user_specific_id = context.grag_id
        all_kbs = context.metadata.get("all_kbs", [])

        logger.info(f"[{context.grag_id}] 🤖 AutoQueryAgent 开始执行")
        logger.info(f"[{context.grag_id}] 🎯 用户指定ID: {user_specific_id}, 可用知识库数: {len(all_kbs)}")

        # ============================================================
        # 步骤 1: 优先查询用户指定的知识库
        # ============================================================
        if user_specific_id and user_specific_id != "default":
            logger.info(f"[{user_specific_id}] 🔍 步骤1: 尝试查询用户指定的知识库")

            # 创建临时上下文，仅包含当前指定的知识库ID
            temp_context = AgentContext(
                grag_id=user_specific_id,
                question=context.question,
                conversation_history=context.conversation_history,
                metadata=context.metadata.copy()
            )

            # 执行查询
            result = await self._execute_query_logic(temp_context)

            # 检查结果有效性
            if result.get("success"):
                answer = result.get("answer") or result.get("final_answer", "")
                # 简单判断答案有效性：长度大于20字符
                if answer and len(answer) > 20:
                    execution_time = time.time() - start_time
                    logger.info(f"[{user_specific_id}] ✅ 指定知识库查询成功，直接返回")
                    return AgentResult(
                        success=True,
                        data={
                            "question": context.question,
                            "answer": answer,
                            "method_used": result.get("method", "unknown"),
                            "grag_id": user_specific_id
                        },
                        message=f"在指定知识库中查询成功",
                        execution_time=execution_time,
                        metadata={"method": result.get("method"), "kb_name": "用户指定知识库"}
                    )

            # 如果指定知识库没查到有效结果，记录日志并准备回退
            logger.warning(f"[{user_specific_id}] ⚠️ 指定知识库查询无有效结果，开始回退查询其他知识库...")

        # ============================================================
        # 步骤 2: 回退逻辑 - 查询其他知识库
        # ============================================================

        # 过滤掉已经查询过的那个（如果有的话）
        fallback_kbs = [kb for kb in all_kbs if kb.get("grag_id") != user_specific_id]

        if not fallback_kbs:
            logger.info(f"[{context.grag_id}] 📭 没有其他可回退的知识库")
            return AgentResult(
                success=False,
                data=None,
                message="指定知识库无答案，且没有其他可用知识库",
                error="no_fallback_kbs",
                execution_time=time.time() - start_time
            )

        logger.info(f"[{context.grag_id}] 🔄 步骤2: 开始在 {len(fallback_kbs)} 个其他知识库中轮询")

        # 复用现有的多知识库查询逻辑，但只传入回退列表
        return await self._query_multiple_kbs(context, fallback_kbs, start_time)

    async def _execute_query_logic(self, context: AgentContext) -> dict:
        """
        内部方法：根据上下文执行单次查询（含复杂度分析和方法选择）
        """
        try:
            # 1. 分析复杂度
            complexity = await self._analyze_complexity(context.question)

            # 2. 选择方法
            method = self._select_method(complexity, context)

            # 3. 执行查询
            if method == "tog":
                result = self.tog_tool.execute(
                    grag_id=context.grag_id,
                    question=context.question,
                    max_depth=context.metadata.get("max_depth", 5),
                    max_width=context.metadata.get("max_width", 5)
                )
            elif method == "graphrag":
                result = self.graphrag_tool.execute(
                    grag_id=context.grag_id,
                    question=context.question,
                    method=context.metadata.get("graphrag_method", "local")
                )
            else:  # hybrid
                result = await self.hybrid_tool.execute(
                    grag_id=context.grag_id,
                    question=context.question,
                    max_depth=context.metadata.get("max_depth", 5),
                    max_width=context.metadata.get("max_width", 5),
                    method=context.metadata.get("graphrag_method", "local")
                )

            # 将选择的方法附加到结果中，方便上层判断
            result["method"] = method
            return result

        except Exception as e:
            logger.error(f"查询执行异常: {e}")
            return {"success": False, "error": str(e)}

    async def _query_multiple_kbs(self, context: AgentContext, kb_list: list, start_time: float) -> AgentResult:
        """
        遍历知识库列表进行查询（回退查询专用）
        """
        # 分析一次复杂度即可
        complexity = await self._analyze_complexity(context.question)

        for kb in kb_list:
            kb_grag_id = kb.get("graph_key")
            kb_name = kb.get("name", "未知")

            logger.info(f"[{kb_grag_id}] 🔍 尝试回退查询知识库: {kb_name}")

            # 更新上下文的 grag_id
            temp_context = AgentContext(
                grag_id=kb_grag_id,
                question=context.question,
                metadata=context.metadata
            )

            # 执行查询
            result = await self._execute_query_logic(temp_context)

            # 如果查询成功且结果有效，返回
            if result.get("success"):
                answer = result.get("answer") or result.get("final_answer", "")
                if answer and len(answer) > 20:
                    execution_time = time.time() - start_time
                    logger.info(f"[{kb_grag_id}] ✅ 回退查询成功，在知识库 '{kb_name}' 找到答案")

                    return AgentResult(
                        success=True,
                        data={
                            "question": context.question,
                            "answer": answer,
                            "method_used": result.get("method"),
                            "complexity": complexity,
                            "grag_id": kb_grag_id,
                        },
                        message=f"查询成功 (知识库: {kb_name})",
                        execution_time=execution_time,
                        metadata={
                            "method": result.get("method"),
                            "complexity": complexity,
                            "kb_name": kb_name,
                            "is_fallback": True  # 标记这是回退结果
                        }
                    )

        # 所有回退知识库都查完了，没找到
        execution_time = time.time() - start_time
        kb_names = ", ".join([kb.get("name", "未知") for kb in kb_list])
        return AgentResult(
            success=False,
            data=None,
            message=f"在指定知识库及 {len(kb_list)} 个备用知识库中均未找到相关答案。已尝试: {kb_names}",
            error="no_answer_found_in_any_kb",
            execution_time=execution_time
        )

    # 保留原有的辅助方法
    async def _analyze_complexity(self, question: str) -> str:
        prompt = f"""分析问题复杂度：{question}\n返回: simple / moderate / complex"""
        try:
            response = await llm_client.generate(prompt, temperature=0.1)
            if "simple" in response.lower(): return "simple"
            elif "complex" in response.lower(): return "complex"
            return "moderate"
        except:
            return "moderate"

    def _select_method(self, complexity: str, context: AgentContext) -> str:
        query_lower = context.question.lower()
        if "tog" in query_lower or "思维图" in query_lower: return "tog"
        if "graphrag" in query_lower: return "graphrag"
        if "混合" in query_lower or "hybrid" in query_lower: return "hybrid"

        if complexity == "simple": return "graphrag"
        if complexity == "complex": return "hybrid"
        return "tog"