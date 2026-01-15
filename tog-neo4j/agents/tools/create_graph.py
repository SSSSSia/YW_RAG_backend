"""
创建知识图谱工具
"""
from typing import Dict, Any
from pathlib import Path
from services.graph_creation_service import GraphCreationService
from utils.logger import logger


class CreateGraphTool:
    """创建知识图谱工具"""

    def __init__(self):
        self.name = "create_graph"
        self.description = "创建新的知识图谱"

    async def execute(self, grag_id: str, file_path: str, filename: str) -> Dict[str, Any]:
        """执行图谱创建"""
        try:
            logger.info(f"[{grag_id}] 🔧 工具: 创建知识图谱")

            service = GraphCreationService(grag_id)
            await service.create_graph(file_path, filename)

            return {
                "success": True,
                "message": "图谱创建任务已启动",
                "grag_id": grag_id
            }
        except Exception as e:
            logger.error(f"创建图谱失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }