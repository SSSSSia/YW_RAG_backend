"""
基于 LangChain 的工具实现
"""
from typing import Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from services.query_graphrag_service import GraphRAGService
from services.query_tog_service import ToGService
from services.query_hybrid_service import HybridQueryService
from utils.logger import logger


# ==================== 输入模型定义 ====================

class GraphRAGInput(BaseModel):
    """GraphRAG 查询输入"""
    grag_id: str = Field(description="知识库ID")
    question: str = Field(description="用户问题")
    method: str = Field(default="local", description="查询方法: local 或 global")


class ToGInput(BaseModel):
    """ToG 查询输入"""
    grag_id: str = Field(description="知识库ID")
    question: str = Field(description="用户问题")
    max_depth: int = Field(default=5, description="最大推理深度")
    max_width: int = Field(default=5, description="最大推理宽度")


class HybridInput(BaseModel):
    """混合查询输入"""
    grag_id: str = Field(description="知识库ID")
    question: str = Field(description="用户问题")
    max_depth: int = Field(default=10, description="最大推理深度")
    max_width: int = Field(default=3, description="最大推理宽度")
    method: str = Field(default="local", description="GraphRAG方法")


# ==================== Tool 实现 ====================

class GraphRAGTool(BaseTool):
    """GraphRAG 查询工具"""
    name: str = "graphrag_query"
    description: str = """
    使用 GraphRAG 方法查询知识图谱。
    适用于简单的事实查询和信息检索。
    输入参数:
    - grag_id: 知识库ID
    - question: 用户问题
    - method: 查询方法 (local)
    """
    args_schema: Type[BaseModel] = GraphRAGInput

    def _run(
        self,
        grag_id: str,
        question: str,
        method: str = "local",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """同步执行"""
        try:
            logger.info(f"[GraphRAGTool] 查询知识库: {grag_id}")
            service = GraphRAGService(grag_id)
            success, answer, exec_time = service.query(question, method)

            if success:
                return answer
            else:
                return f"查询失败: {answer}"
        except Exception as e:
            logger.error(f"GraphRAG工具执行失败: {e}")
            return f"工具执行错误: {str(e)}"

    async def _arun(
        self,
        grag_id: str,
        question: str,
        method: str = "local",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """异步执行"""
        return self._run(grag_id, question, method, run_manager)


class ToGTool(BaseTool):
    """ToG (Think-on-Graph) 查询工具"""
    name: str = "tog_query"
    description: str = """
    使用 ToG (Think-on-Graph) 方法进行推理查询。
    适用于需要多步推理和逻辑链的复杂问题。
    输入参数:
    - grag_id: 知识库ID
    - question: 用户问题
    - max_depth: 最大推理深度
    - max_width: 最大推理宽度
    """
    args_schema: Type[BaseModel] = ToGInput

    def _run(
        self,
        grag_id: str,
        question: str,
        max_depth: int = 5,
        max_width: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """同步执行"""
        try:
            logger.info(f"[ToGTool] ToG推理查询: {grag_id}")
            service = ToGService(grag_id, max_depth, max_width)
            result = service.reason(question)

            if result.get("success"):
                answer = result.get("answer", "")
                return answer
            else:
                return result.get('error', '未知错误')
        except Exception as e:
            logger.error(f"ToG工具执行失败: {e}")
            return f"工具执行错误: {str(e)}"

    async def _arun(
        self,
        grag_id: str,
        question: str,
        max_depth: int = 5,
        max_width: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """异步执行"""
        return self._run(grag_id, question, max_depth, max_width, run_manager)


class HybridQueryTool(BaseTool):
    """混合查询工具"""
    name: str = "hybrid_query"
    description: str = """
    使用 ToG + GraphRAG 混合方法查询知识图谱。
    适用于需要深度推理和广泛检索的复杂问题。
    输入参数:
    - grag_id: 知识库ID
    - question: 用户问题
    - max_depth: 最大推理深度
    - max_width: 最大推理宽度
    - method: GraphRAG方法
    """
    args_schema: Type[BaseModel] = HybridInput

    def _run(
        self,
        grag_id: str,
        question: str,
        max_depth: int = 10,
        max_width: int = 3,
        method: str = "local",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """同步执行 (实际调用异步)"""
        import asyncio
        return asyncio.run(self._arun(grag_id, question, max_depth, max_width, method, run_manager))

    async def _arun(
        self,
        grag_id: str,
        question: str,
        max_depth: int = 10,
        max_width: int = 3,
        method: str = "local",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """异步执行"""
        try:
            logger.info(f"[HybridTool] 混合查询: {grag_id}")
            service = HybridQueryService(grag_id, max_depth, max_width, method)
            result = await service.query(question)

            if result.get("success"):
                answer = result.get("final_answer", "")
                return answer
            else:
                return f"查询失败: {result.get('error', '未知错误')}"
        except Exception as e:
            logger.error(f"混合查询工具执行失败: {e}")
            return f"工具执行错误: {str(e)}"


# ==================== 工具列表 ====================

def get_all_tools():
    """获取所有可用工具"""
    return [
        GraphRAGTool(),
        ToGTool(),
        HybridQueryTool()
    ]
