"""
GraphRAG查询服务
"""
import subprocess
import time
from pathlib import Path
from typing import Tuple
from utils.logger import logger, log_step
from utils.common import run_command_with_progress
from core.config import settings


class GraphRAGService:
    """GraphRAG查询服务"""

    def __init__(self, grag_id: str):
        self.grag_id = grag_id
        self.user_path = Path(settings.graphrag_root) / grag_id

    def query(self, question: str, method: str = "local") -> Tuple[bool, str, float]:
        """执行GraphRAG查询"""
        start_time = time.time()

        if not self.user_path.exists():
            error_msg = f"目录 {self.grag_id} 不存在，请先创建知识图谱"
            logger.error(f"[{self.grag_id}] ❌ {error_msg}")
            return False, error_msg, 0

        logger.info(f"[{self.grag_id}] 执行GraphRAG查询: {question}")

        query_command = (
            f'python -m graphrag query '
            f'--root {self.user_path} '
            f'--method {method} '
            f'--query "{question}"'
        )

        success, stdout, stderr = run_command_with_progress(
            query_command,
            f"GraphRAG {method} 查询",
            self.grag_id
        )

        execution_time = time.time() - start_time

        if success:
            result = stdout.strip()
            logger.info(f"[{self.grag_id}] ✅ 查询成功，耗时: {execution_time:.2f}秒")
            return True, result, execution_time
        else:
            error_msg = stderr[:500] if stderr else "未知错误"
            logger.error(f"[{self.grag_id}] ❌ 查询失败: {error_msg}")
            return False, error_msg, execution_time