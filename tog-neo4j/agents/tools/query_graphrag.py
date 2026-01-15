"""
GraphRAG 查询工具
"""
from typing import Dict, Any
from services.query_graphrag_service import GraphRAGService
from utils.logger import logger


class QueryGraphRAGTool:
    """GraphRAG 查询工具"""

    def __init__(self):
        self.name = "query_graphrag"
        self.description = "使用 GraphRAG 方法查询知识图谱"

    def execute(
            self,
            grag_id: str,
            question: str,
            method: str = "local"
    ) -> Dict[str, Any]:
        """执行 GraphRAG 查询"""
        try:
            logger.info(f"[{grag_id}] 🔧 工具: GraphRAG查询")

            service = GraphRAGService(grag_id)
            success, answer, execution_time = service.query(question, method)

            return {
                "success": success,
                "answer": answer,
                "execution_time": execution_time,
                "method": f"GraphRAG-{method}"
            }
        except Exception as e:
            logger.error(f"GraphRAG查询失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "method": "GraphRAG"
            }