"""
ToG业务逻辑服务
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.logger import logger,log_step
from core.neo4j_db import db_manager
from core.llm_client import llm_client
from scripts.ywretriever import Retriever, entity_linking
from core.config import settings


class ToGService:
    """ToG推理服务"""

    def __init__(self, grag_id: str, max_depth: int = 5, max_width: int = 5):
        self.grag_id = grag_id
        self.max_depth = max_depth
        self.max_width = max_width
        self.neo4j = db_manager.get_connector(grag_id)
        self.retriever = self._init_retriever()
        self.prompts = self._load_prompts()

    def _init_retriever(self):
        """初始化实体链接检索器"""
        try:
            retriever_path = Path(settings.graphrag_root) / self.grag_id / ".retrive"
            retriever = Retriever(
                retrievel_type="dense",
                retriever_version=str(retriever_path)
            )
            logger.info(f"实体链接检索器初始化成功: {retriever_path}")
            return retriever
        except Exception as e:
            logger.warning(f"实体链接检索器初始化失败: {e}, 将使用原有的实体匹配方法")
            return None

    def _load_prompts(self) -> Dict[str, str]:
        """加载提示词模板"""
        return {
            "entity_extraction": """Given the question: "{question}"
            Please extract all key entities mentioned in this question.
            Return only the entity names, separated by commas.
            Entities:""",

            "relation_selection": """Given the question: "{question}"
            Current entities: {entities}
            Available relations: {relations}

            Please select the top {beam_width} most relevant relations that help answer the question.
            Return only the relation names, separated by commas.
            Selected relations:""",

            "entity_selection": """Given the question: "{question}"
            Current relation: {relation}
            Available entities: {entities}

            Please select the top {beam_width} most relevant entities that help answer the question.
            Return only the entity names, separated by commas.
            Selected entities:""",

            "reasoning_evaluation": """Given the question: "{question}"
            Retrieved knowledge paths:
            {paths}

            Can you answer the question with sufficient confidence based on these paths and your knowledge?
            Answer with only "Yes" or "No".
            Answer:""",

            "answer_generation": """Based on the question and retrieved knowledge paths, please provide a clear, Markdown-formatted answer suitable for frontend rendering.

            Requirements:
            1. Use Markdown syntax for better readability (e.g., **bold** key terms, `code` for entities/paths).
            2. Organize the answer into a numbered list (1. 2. 3. ...).
            3. Ensure each point starts on a new line.
            4. Provide a concise and structured response.

            Question: {question}
            Knowledge paths:
            {paths}

            Answer:"""

        }

    def _extract_topic_entities(self, question: str) -> List[str]:
        """从问题中提取主题实体"""
        prompt = self.prompts["entity_extraction"].format(question=question)
        # response = llm_client.chat(prompt)
        response = llm_client.chat_with_siliconflow(prompt)
        raw_entities = [e.strip() for e in response.split(",") if e.strip()]
        logger.info(f"LLM提取的原始实体: {raw_entities}")

        # 使用实体链接模块进行链接
        if self.retriever:
            try:
                linked_entities = entity_linking(
                    retriever_obj=self.retriever,
                    entities=raw_entities,
                    threshold=settings.entity_linking_threshold
                )
                logger.info(f"实体链接后的结果: {linked_entities}")
                matched_entities = list(dict.fromkeys(linked_entities))[:self.max_width]
            except Exception as e:
                logger.error(f"实体链接失败: {e}, 使用原有匹配方法")
                matched_entities = self._fallback_entity_matching(raw_entities)
        else:
            matched_entities = self._fallback_entity_matching(raw_entities)

        logger.info(f"最终提取到的主题实体: {matched_entities}")
        return matched_entities

    def _fallback_entity_matching(self, raw_entities: List[str]) -> List[str]:
        """原有的实体匹配逻辑（备用方案）"""
        matched_entities = []
        for entity in raw_entities:
            exact_matches = self.neo4j.search_entities_exact(entity)
            if exact_matches:
                matched_entities.append(exact_matches[0]["entity_name"])
                continue

            fuzzy_matches = self.neo4j.search_entities_fuzzy(entity, limit=1)
            if fuzzy_matches:
                matched_entities.append(fuzzy_matches[0]["entity_name"])
                continue

            matched_entities.append(entity)

        return list(dict.fromkeys(matched_entities))[:self.max_width]

    def _explore_relations(self, entities: List[str], question: str) -> Dict[str, List[str]]:
        """关系探索"""
        entity_relations = {}

        for entity in entities:
            neighbors = self.neo4j.get_entity_neighbors(entity, depth=1)
            all_relations = list(set([
                n.get("relation") for n in neighbors if n.get("relation")
            ]))

            if not all_relations:
                entity_relations[entity] = []
                continue

            if len(all_relations) <= self.max_width:
                selected_relations = all_relations
            else:
                prompt = self.prompts["relation_selection"].format(
                    question=question,
                    entities=entity,
                    relations=", ".join(all_relations),
                    beam_width=self.max_width
                )
                # response = llm_client.chat(prompt)
                response = llm_client.chat_with_siliconflow(prompt)
                selected_relations = [r.strip() for r in response.split(",") if r.strip()]
                selected_relations = selected_relations[:self.max_width]

            entity_relations[entity] = selected_relations
            logger.info(f"实体 '{entity}' 选择的关系: {selected_relations}")

        return entity_relations

    def _explore_entities(self, entity_relations: Dict[str, List[str]], question: str) -> List[Dict[str, Any]]:
        """实体探索"""
        exploration_results = []

        for source_entity, relations in entity_relations.items():
            for relation in relations:
                neighbors = self.neo4j.get_entity_neighbors(source_entity, depth=1)
                target_candidates = [
                    n.get("target_entity")
                    for n in neighbors
                    if n.get("relation") == relation and n.get("target_entity")
                ]

                if not target_candidates:
                    continue

                target_candidates = list(dict.fromkeys(target_candidates))

                if len(target_candidates) <= self.max_width:
                    selected_targets = target_candidates
                else:
                    prompt = self.prompts["entity_selection"].format(
                        question=question,
                        relation=relation,
                        entities=", ".join(target_candidates[:20]),
                        beam_width=self.max_width
                    )
                    # response = llm_client.chat(prompt)
                    response = llm_client.chat_with_siliconflow(prompt)
                    selected_targets = [e.strip() for e in response.split(",") if e.strip()]
                    selected_targets = selected_targets[:self.max_width]

                exploration_results.append({
                    "source": source_entity,
                    "relation": relation,
                    "targets": selected_targets
                })

                logger.info(f"路径: {source_entity} --[{relation}]--> {selected_targets}")

        return exploration_results

    def _beam_search_iteration(self, current_paths: List[List[Dict]], question: str) -> List[List[Dict]]:
        """执行一次beam search迭代"""
        if not current_paths:
            topic_entities = self._extract_topic_entities(question)
            current_paths = [[{"source": None, "relation": None, "target": e}] for e in topic_entities]

        tail_entities = [path[-1]["target"] for path in current_paths]
        entity_relations = self._explore_relations(tail_entities, question)
        exploration_results = self._explore_entities(entity_relations, question)

        new_paths = []
        for path in current_paths:
            tail_entity = path[-1]["target"]
            for result in exploration_results:
                if result["source"] == tail_entity:
                    for target in result["targets"]:
                        new_path = path + [{
                            "source": result["source"],
                            "relation": result["relation"],
                            "target": target
                        }]
                        new_paths.append(new_path)

        return new_paths[:self.max_width] if new_paths else current_paths

    def _format_paths(self, paths: List[List[Dict]]) -> str:
        """格式化路径用于显示"""
        formatted = []
        for i, path in enumerate(paths, 1):
            path_str = " -> ".join([
                f"({step['source']}) --[{step['relation']}]--> ({step['target']})"
                for step in path if step['relation'] is not None
            ])
            if path_str:
                formatted.append(f"Path {i}: {path_str}")
        return "\n".join(formatted)

    def _evaluate_sufficiency(self, question: str, paths: List[List[Dict]]) -> bool:
        """评估当前路径是否足以回答问题"""
        if not paths or not any(paths):
            return False

        paths_text = self._format_paths(paths)
        prompt = self.prompts["reasoning_evaluation"].format(
            question=question,
            paths=paths_text
        )

        # response = llm_client.chat(prompt).lower()
        response = llm_client.chat_with_siliconflow(prompt).lower()
        return "yes" in response

    def _generate_answer(self, question: str, paths: List[List[Dict]]) -> str:
        """基于路径生成答案"""
        paths_text = self._format_paths(paths)
        prompt = self.prompts["answer_generation"].format(
            question=question,
            paths=paths_text
        )

        return llm_client.chat_with_siliconflow(prompt, temperature=0.1)

    def reason(self, question: str) -> Dict[str, Any]:
        """ToG主推理流程"""
        start_time = time.time()

        try:
            logger.info(f"[{self.grag_id}] 开始ToG推理 - 问题: {question}")
            logger.info(f"参数: max_depth={self.max_depth}, beam_width={self.max_width}")

            current_paths = []

            for depth in range(self.max_depth):
                logger.info(f"========== 深度 {depth + 1}/{self.max_depth} ==========")
                current_paths = self._beam_search_iteration(current_paths, question)

                if not current_paths:
                    logger.warning(f"深度 {depth + 1}: 无法继续探索")
                    break

                if self._evaluate_sufficiency(question, current_paths):
                    logger.info(f"深度 {depth + 1}: 信息充足,开始生成答案")
                    break

            if current_paths:
                answer = self._generate_answer(question, current_paths)
                success = True
            else:
                answer = "无法从知识图谱中找到足够的信息来回答该问题。"
                success = False

            execution_time = time.time() - start_time

            return {
                "success": success,
                "question": question,
                "answer": answer,
                "execution_time": execution_time,
                "error_message": None
            }

        except Exception as e:
            logger.error(f"推理过程出错: {e}", exc_info=True)
            return {
                "success": False,
                "question": question,
                "answer": f"推理过程出错: {str(e)}",
                "execution_time": time.time() - start_time,
                "error_message": str(e)
            }