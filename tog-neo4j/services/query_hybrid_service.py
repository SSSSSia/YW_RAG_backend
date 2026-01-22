"""
混合查询业务逻辑
"""
import time
from pathlib import Path
from utils.logger import logger,log_step
from services.query_tog_service import ToGService
from services.query_graphrag_service import GraphRAGService
from core.llm_client import llm_client


class HybridQueryService:
    """ToG+GraphRAG混合查询服务"""

    def __init__(self, grag_id: str, max_depth: int = 5, max_width: int = 5, method: str = "local"):
        self.grag_id = grag_id
        self.max_depth = max_depth
        self.max_width = max_width
        self.method = method

    async def query(self, question: str) -> dict:
        """执行混合查询"""
        start_time = time.time()

        logger.info(f"[{self.grag_id}] 🔍 开始ToG+GraphRAG混合查询")

        # 执行ToG查询
        log_step(1, 4, "执行ToG查询", self.grag_id)
        tog_service = ToGService(self.grag_id, self.max_depth, self.max_width)
        tog_result = tog_service.reason(question)
        tog_answer = tog_result.get("answer", "")
        tog_success = tog_result.get("success", False)

        # 执行GraphRAG查询
        log_step(2, 4, "执行GraphRAG查询", self.grag_id)
        graphrag_service = GraphRAGService(self.grag_id)
        graph_success, graph_answer, _ = graphrag_service.query(question, self.method)

        # 整合答案
        log_step(3, 4, "整合两个答案", self.grag_id)
        if not tog_answer and not graph_answer:
            return {
                "success": False,
                "question": question,
                "error": "两种查询方法都未返回有效答案"
            }

        integration_prompt = f"""你是一个专业的知识图谱查询助手。我使用两种不同的方法查询了同一个问题，现在需要你整合两个答案，给出最准确、最全面的回答。

        **问题：** {question}

        **方法1 - ToG（思维图谱）的答案：**
        {tog_answer if tog_answer else "(未获取到答案)"}

        **方法2 - GraphRAG的答案：**
        {graph_answer if graph_answer else "(未获取到答案)"}

        请综合以上两个答案，给出一个最终答案。要求：
        1. **内容整合**：综合两个答案的优点和补充信息，去重并确保准确性、完整性。
        2. **格式要求**：请使用 **Markdown** 格式进行美化，规范如下：
           - 使用 `###` 标记主要的小节标题。
           - 关键实体、核心结论或重要数据请使用 `**粗体**` 强调。
           - 使用 `1.` 或 `-` 列表形式清晰地分点陈述。
        3. **冲突处理**：如果两个答案存在冲突或矛盾，请务必在文末增加一个 `### ⚠️ 答案差异与说明` 的独立章节，详细列出冲突点并说明你的判断依据。
        4. **结构清晰**：确保输出内容层级分明，便于前端直接渲染展示。

        **最终答案：**"""

        log_step(4, 4, "使用大模型生成最终答案", self.grag_id)
        try:
            final_answer = await llm_client.generate(integration_prompt)
        except Exception as e:
            logger.error(f"整合答案生成失败: {e}")
            final_answer = tog_answer if len(tog_answer) > len(graph_answer) else graph_answer

        execution_time = time.time() - start_time

        return {
            "success": True,
            "question": question,
            "final_answer": final_answer,
            "tog_answer": tog_answer,
            "graphrag_answer": graph_answer,
            "execution_time": execution_time
        }