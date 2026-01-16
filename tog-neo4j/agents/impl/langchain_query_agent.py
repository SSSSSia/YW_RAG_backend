import asyncio
from typing import Dict, Any, List

# LangChain Core Imports (LangChain 0.1+ 规范)
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama  # 使用 Ollama 替代 OpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 引入你现有的工具类
from agents.tools.query_hybrid import QueryHybridTool
from agents.tools.query_graphrag import QueryGraphRAGTool
from agents.tools.query_tog import QueryToGTool
from core.config import settings
from utils.logger import logger
from utils.java_backend import get_knowledge_bases


class LangChainQueryAgent:
    """
    基于 LangChain 1.0 实现的图谱查询 Agent
    集成了 Hybrid, GraphRAG, ToG 三种工具
    """

    def __init__(self, model_name: str = None, temperature: float = 0):
        # 1. 初始化原始工具类
        self._hybrid_tool = QueryHybridTool()
        self._graphrag_tool = QueryGraphRAGTool()
        self._tog_tool = QueryToGTool()

        # 2. 初始化 LLM (使用 Ollama)
        model = model_name or settings.llm_model
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.llm_api_url,
            timeout=120
        )

        # 3. 构建 LangChain 工具集
        self.tools = self._build_langchain_tools()

        # 4. 构建 Agent
        self.agent_executor = self._build_agent_executor()

    def _build_langchain_tools(self) -> List[Any]:
        """
        将现有的工具类方法封装为 LangChain Tools
        """

        # --- 封装 Hybrid 查询 (Async) ---
        @tool("hybrid_query")
        async def hybrid_query(grag_id: str, question: str, max_depth: int = 5, max_width: int = 5) -> Dict[str, Any]:
            """
            当需要综合查询或者问题比较复杂需要多跳推理时使用此工具。
            结合了 ToG (Think-on-Graph) 和 GraphRAG 的优势。
            """
            # 直接调用原始类的 async execute
            return await self._hybrid_tool.execute(
                grag_id=grag_id,
                question=question,
                max_depth=max_depth,
                max_width=max_width
            )

        # --- 封装 GraphRAG 查询 (Sync) ---
        @tool("graphrag_query")
        def graphrag_query(grag_id: str, question: str, method: str = "local") -> Dict[str, Any]:
            """
            当需要进行全局摘要、社区发现或宏观理解图谱内容时使用此工具。
            Method 可以是 'local' (局部) 或 'global' (全局)。
            """
            return self._graphrag_tool.execute(
                grag_id=grag_id,
                question=question,
                method=method
            )

        # --- 封装 ToG 查询 (Sync) ---
        @tool("tog_query")
        def tog_query(grag_id: str, question: str, max_depth: int = 5, max_width: int = 5) -> Dict[str, Any]:
            """
            当需要沿着图谱路径进行深度推理、寻找实体间特定关系链时使用此工具。
            ToG 代表 Think-on-Graph。
            """
            return self._tog_tool.execute(
                grag_id=grag_id,
                question=question,
                max_depth=max_depth,
                max_width=max_width
            )

        return [hybrid_query, graphrag_query, tog_query]

    def _build_agent_executor(self) -> AgentExecutor:
        """构建 Agent 执行器"""

        # 定义 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的知识图谱查询助手。根据用户的问题和提供的 grag_id（图谱ID），"
                       "选择最合适的工具（Hybrid, GraphRAG, 或 ToG）来回答问题。\n"
                       "1. 如果问题需要宏观理解或总结，优先用 GraphRAG。\n"
                       "2. 如果问题需要多步推理，优先用 ToG。\n"
                       "3. 如果不确定或需要综合能力，使用 Hybrid。\n"
                       "请确保在调用工具时传入正确的 grag_id。"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 绑定工具到 LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # 创建 Agent (使用 Tool Calling 模式，现代 LLM 的标准做法)
        agent = create_tool_calling_agent(llm_with_tools, self.tools, prompt)

        # 创建执行器
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def query(self, grag_id: str, question: str) -> Dict[str, Any]:
        """
        对外暴露的统一查询接口
        """
        try:
            # 如果 grag_id 为空或为 default，则获取所有知识库并循环查询
            if not grag_id or grag_id == "default":
                logger.info("📚 grag_id 为空，开始获取知识库列表并循环查询")
                
                # 获取知识库列表
                knowledge_bases = await get_knowledge_bases()
                
                if not knowledge_bases:
                    logger.warning("❌ 未找到任何知识库")
                    return {
                        "success": False,
                        "error": "未找到任何可用的知识库"
                    }
                
                logger.info(f"✅ 获取到 {len(knowledge_bases)} 个知识库: {[kb['name'] for kb in knowledge_bases]}")
                
                # 遍历知识库进行查询
                for kb in knowledge_bases:
                    kb_grag_id = kb.get("graph_key") or kb.get("grag_id")
                    kb_name = kb.get("name", "未知知识库")
                    
                    logger.info(f"[{kb_grag_id}] 🔍 尝试查询知识库: {kb_name}")
                    
                    # 构造输入，显式包含 grag_id 以便 Agent 可以在上下文中理解
                    input_text = f"Current Graph ID: {kb_grag_id}\nQuestion: {question}"

                    try:
                        # 使用 ainvoke 因为其中包含异步工具 (Hybrid)
                        response = await self.agent_executor.ainvoke({
                            "input": input_text
                        })
                        
                        result = response["output"]
                        # 检查结果是否有效（例如，答案长度大于一定阈值）
                        if result and len(result.strip()) > 20:  # 简单的有效性检查
                            logger.info(f"[{kb_grag_id}] ✅ 在知识库 '{kb_name}' 中找到有效答案")
                            
                            return {
                                "success": True,
                                "result": result,
                                "details": response,  # 包含中间步骤
                                "kb_used": kb_name,
                                "kb_grag_id": kb_grag_id
                            }
                        else:
                            logger.info(f"[{kb_grag_id}] ⚠️ 知识库 '{kb_name}' 查询结果无效，继续尝试下一个")
                            continue
                    except Exception as kb_error:
                        logger.error(f"[{kb_grag_id}] ❌ 查询知识库 '{kb_name}' 时出错: {kb_error}")
                        continue
                
                # 所有知识库都查询失败
                logger.error("❌ 所有知识库查询都失败")
                return {
                    "success": False,
                    "error": "在所有知识库中都未能找到满意答案"
                }
            else:
                # 原有逻辑：指定知识库查询
                logger.info(f"Agent 收到查询: [{grag_id}] {question}")

                # 构造输入，显式包含 grag_id 以便 Agent 可以在上下文中理解
                input_text = f"Current Graph ID: {grag_id}\nQuestion: {question}"

                # 使用 ainvoke 因为其中包含异步工具 (Hybrid)
                response = await self.agent_executor.ainvoke({
                    "input": input_text
                })

                return {
                    "success": True,
                    "result": response["output"],
                    "details": response  # 包含中间步骤
                }
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# --- 使用示例 (可以在 main.py 中调用) ---
if __name__ == "__main__":
    # 模拟运行
    async def main():
        agent = LangChainQueryAgent()

        # 示例：Agent 应该会选择 GraphRAG 或 Hybrid
        result = await agent.query(
            grag_id="test_graph_001",
            question="总结一下这个知识图谱里关于人工智能的主要观点"
        )
        print("Final Answer:", result.get("result"))


    # 运行异步主函数
    asyncio.run(main())