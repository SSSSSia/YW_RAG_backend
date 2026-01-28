"""
ToG 查询工具
"""
from typing import Dict, Any
from services.query_tog_service import ToGService
from utils.logger import logger


class QueryToGTool:
    """ToG 查询工具"""

    def __init__(self):
        self.name = "query_tog"
        self.description = "使用 ToG (Think-on-Graph) 方法查询知识图谱"

    def execute(
            self,
            grag_id: str,
            question: str,
            max_depth: int = 3,
            max_width: int = 3
    ) -> Dict[str, Any]:
        """执行 ToG 查询"""
        try:
            logger.info(f"[{grag_id}] 🔧 工具: ToG查询")

            service = ToGService(grag_id, max_depth, max_width)
            result = service.reason(question)

            return {
                "success": result.get("success", False),
                "answer": result.get("answer", ""),
                "execution_time": result.get("execution_time", 0),
                "method": "ToG"
            }
        except Exception as e:
            logger.error(f"ToG查询失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "method": "ToG"
            }