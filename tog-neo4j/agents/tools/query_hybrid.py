"""
混合查询工具
"""
from typing import Dict, Any
from services.query_hybrid_service import HybridQueryService
from utils.logger import logger


class QueryHybridTool:
    """混合查询工具"""

    def __init__(self):
        self.name = "query_hybrid"
        self.description = "使用 ToG + GraphRAG 混合方法查询知识图谱"

    async def execute(
            self,
            grag_id: str,
            question: str,
            max_depth: int = 3,
            max_width: int = 3,
            method: str = "local"
    ) -> Dict[str, Any]:
        """执行混合查询"""
        try:
            logger.info(f"[{grag_id}] 🔧 工具: 混合查询")

            service = HybridQueryService(grag_id, max_depth, max_width, method)
            result = await service.query(question)

            return result
        except Exception as e:
            logger.error(f"混合查询失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "method": "Hybrid"
            }