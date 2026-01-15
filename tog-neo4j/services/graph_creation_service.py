"""
图谱创建业务逻辑
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from utils.logger import logger, log_step
from utils.common import run_command_with_progress
from utils.java_backend import notify_java_backend
from core.config import settings
from scripts.deal_graph import main as deal_graph_main
from scripts.insert_to_neo4j import main as insert_neo4j_main
from scripts.ywretriever import crtDenseRetriever
from core.database import db_manager
import sys



class GraphCreationService:
    """图谱创建服务"""

    def __init__(self, grag_id: str):
        self.grag_id = grag_id
        self.user_path = Path(settings.graphrag_root) / grag_id
        self.input_dir = self.user_path / "input"
        self.output_dir = self.user_path / "output"

    async def create_graph(self, file_path: str, filename: str):
        """执行图谱创建的完整流程"""
        try:
            current_python = sys.executable
            logger.info(f"[{self.grag_id}] 使用解释器: {current_python}")
            logger.info(f"[{self.grag_id}] 📄 开始后台图谱创建任务")
            TOTAL_STEPS = 7

            # 步骤1: 初始化GraphRAG
            log_step(1, TOTAL_STEPS, "初始化GraphRAG配置", self.grag_id)
            # init_command = f"python -m graphrag init --root {self.user_path}"
            init_command = f'"{current_python}" -m graphrag init --root "{self.user_path}"'


            success, stdout, stderr = run_command_with_progress(
                init_command, "GraphRAG初始化", self.grag_id
            )

            if not success:
                await notify_java_backend(
                    graph_key=self.grag_id,
                    code=500,
                    build_message="初始化失败",
                )
                return

            # 步骤2: 复制配置文件
            log_step(2, TOTAL_STEPS, "配置settings.yaml", self.grag_id)
            user_settings_path = self.user_path / "settings.yaml"
            if Path(settings.base_settings_path).exists():
                shutil.copy2(settings.base_settings_path, user_settings_path)
                logger.info(f"[{self.grag_id}] ✅ 配置文件已复制")

            # 步骤3: 构建索引
            log_step(3, TOTAL_STEPS, "构建知识图谱索引", self.grag_id)
            index_command = f"python -m graphrag index --root {self.user_path}"

            success, stdout, stderr = run_command_with_progress(
                index_command, "索引构建", self.grag_id
            )

            if not success:
                await notify_java_backend(
                    graph_key=self.grag_id,
                    code=500,
                    build_message="索引构建失败",
                )
                return

            # 步骤4: 提取三元组
            log_step(4, TOTAL_STEPS, "提取三元组数据", self.grag_id)
            extracted_json_path = deal_graph_main(
                input_dir=str(self.output_dir),
                grag_id=self.grag_id
            )

            if not extracted_json_path:
                await notify_java_backend(
                    graph_key=self.grag_id,
                    code=500,
                    build_message="图谱创建成功，但三元组提取失败",
                )
                return

            # 步骤5: 导入数据到Neo4j
            log_step(5, TOTAL_STEPS, "导入数据到Neo4j数据库", self.grag_id)
            import_success = insert_neo4j_main(json_file=extracted_json_path)

            if not import_success:
                await notify_java_backend(
                    graph_key=self.grag_id,
                    code=500,
                    build_message="图谱创建成功，但数据库导入失败",
                )
                return

            # 步骤6: 导出节点到CSV
            log_step(6, TOTAL_STEPS, "导出节点到CSV文件", self.grag_id)
            export_success = self.export_nodes_to_csv()

            # 步骤7: 创建密集索引
            log_step(7, TOTAL_STEPS, "根据csv文件建立密集索引", self.grag_id)
            retriv_dir = crtDenseRetriever(
                retriv_dir=str(self.user_path / ".retrive"),
                file_path=str(self.user_path / "nodes_pandas.csv")
            )

            if retriv_dir:
                logger.info(f"[{self.grag_id}] ✅ 索引创建成功: {retriv_dir}")

            # 全部成功，通知Java后端
            logger.info(f"[{self.grag_id}] 🎉 全流程完成！")
            await notify_java_backend(
                graph_key=self.grag_id,
                code=200,
                build_message="知识图谱构建、提取、导入及导出全部完成",
            )

        except Exception as e:
            logger.error(f"[{self.grag_id}] ❌ 后台任务异常: {e}", exc_info=True)
            await notify_java_backend(
                graph_key=self.grag_id,
                code=500,
                build_message="处理过程中发生异常",
            )

    def export_nodes_to_csv(self) -> bool:
        """导出节点到CSV文件"""
        try:
            logger.info(f"[{self.grag_id}] 📤 开始导出节点到CSV")

            connector = db_manager.get_connector(self.grag_id)
            query = """
            MATCH (n)
            WHERE n.grag_id = $grag_id
            RETURN elementId(n) AS id, COALESCE(n.name, '') AS name
            """

            with connector.driver.session() as session:
                result = session.run(query, {"grag_id": self.grag_id})
                nodes_data = [record.data() for record in result]

                if not nodes_data:
                    logger.warning(f"[{self.grag_id}] ⚠️ 数据库中没有匹配该grag_id的节点数据")
                    return False

                import pandas as pd
                df = pd.DataFrame(nodes_data)
                csv_path = self.user_path / "nodes_pandas.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')

                logger.info(f"[{self.grag_id}] ✅ 节点导出完成: {csv_path} ({len(nodes_data)} 个节点)")
                return True

        except Exception as e:
            logger.error(f"[{self.grag_id}] ❌ 导出节点到CSV失败: {e}", exc_info=True)
            return False