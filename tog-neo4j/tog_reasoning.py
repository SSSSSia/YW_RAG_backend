import time
import logging
from typing import List, Dict, Any, Optional
import ollama
from neo4j_connector import Neo4jConnector
from ywretriever import Retriever, entity_linking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToGReasoning:
    """
    ToG (Think-on-Graph) 推理引擎
    实现论文中的三阶段推理流程:
    1. Initialization: 提取主题实体
    2. Exploration: 迭代探索关系和实体 (Beam Search)
    3. Reasoning: 基于探索路径生成答案
    """

    def __init__(
            self,
            neo4j_connector: Neo4jConnector,
            llm_model: str,
            api_key: str,
            beam_width: int = 3,
            max_depth: int = 10,
            retriever_path: str =None,
            entity_linking_threshold: float = 15.0
    ):
        self.neo4j = neo4j_connector
        self.llm_model = llm_model
        self.api_key = api_key
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.prompts = self._load_prompts()

        # 初始化实体链接检索器
        try:
            self.retriever = Retriever(
                retrievel_type="dense",
                retriever_version=retriever_path
            )
            self.entity_linking_threshold = entity_linking_threshold
            logger.info(f"实体链接检索器初始化成功: {retriever_path}")
        except Exception as e:
            logger.warning(f"实体链接检索器初始化失败: {e}, 将使用原有的实体匹配方法")
            self.retriever = None

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

            "answer_generation": """Based on the question and retrieved knowledge paths, please provide a clear answer.

            Requirements:
            1. Organize the answer into numbered points (1. 2. 3. ...)
            2. Each point should start on a new line
            3. Provide a concise and structured response

            Question: {question}
            Knowledge paths:
            {paths}

            Answer:"""
        }

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """调用LLM"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 3000
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    # ============================================================
    # Phase 1: Initialization - 提取主题实体
    # ============================================================

    def _extract_topic_entities(self, question: str) -> List[str]:
        """
        从问题中提取主题实体
        新增：使用 ywretriever 进行实体链接
        返回在知识图谱中匹配到的实体列表
        """
        # Step 1: 使用LLM提取原始实体
        prompt = self.prompts["entity_extraction"].format(question=question)
        response = self._call_llm(prompt)
        raw_entities = [e.strip() for e in response.split(",") if e.strip()]

        logger.info(f"LLM提取的原始实体: {raw_entities}")

        # Step 2: 使用实体链接模块进行链接
        if self.retriever:
            try:
                linked_entities = entity_linking(
                    retriever_obj=self.retriever,
                    entities=raw_entities,
                    threshold=self.entity_linking_threshold
                )
                logger.info(f"实体链接后的结果: {linked_entities}")

                # 去重并限制数量
                matched_entities = list(dict.fromkeys(linked_entities))[:self.beam_width]

            except Exception as e:
                logger.error(f"实体链接失败: {e}, 使用原有匹配方法")
                matched_entities = self._fallback_entity_matching(raw_entities)
        else:
            # 如果检索器未初始化，使用原有的匹配方法
            matched_entities = self._fallback_entity_matching(raw_entities)

        logger.info(f"最终提取到的主题实体: {matched_entities}")
        return matched_entities

    def _fallback_entity_matching(self, raw_entities: List[str]) -> List[str]:
        """
        原有的实体匹配逻辑（作为备用方案）
        """
        matched_entities = []
        for entity in raw_entities:
            # 尝试精确匹配
            exact_matches = self.neo4j.search_entities_exact(entity)
            if exact_matches:
                matched_entities.append(exact_matches[0]["entity_name"])
                continue

            # 尝试模糊匹配
            fuzzy_matches = self.neo4j.search_entities_fuzzy(entity, limit=1)
            if fuzzy_matches:
                matched_entities.append(fuzzy_matches[0]["entity_name"])
                continue

            # 如果都没匹配到,保留原始实体
            matched_entities.append(entity)

        # 去重并限制数量
        return list(dict.fromkeys(matched_entities))[:self.beam_width]

    # ============================================================
    # Phase 2: Exploration - Beam Search探索
    # ============================================================

    def _explore_relations(
            self,
            entities: List[str],
            question: str
    ) -> Dict[str, List[str]]:
        """
        关系探索:为每个实体找到最相关的关系
        返回: {entity: [selected_relations]}
        """
        entity_relations = {}

        for entity in entities:
            # Step 1: Search - 获取所有相关关系
            neighbors = self.neo4j.get_entity_neighbors(entity, depth=1)

            # 提取所有关系类型
            all_relations = list(set([
                n.get("relation") for n in neighbors
                if n.get("relation")
            ]))

            if not all_relations:
                entity_relations[entity] = []
                continue

            # Step 2: Prune - 使用LLM选择最相关的关系
            if len(all_relations) <= self.beam_width:
                selected_relations = all_relations
            else:
                prompt = self.prompts["relation_selection"].format(
                    question=question,
                    entities=entity,
                    relations=", ".join(all_relations),
                    beam_width=self.beam_width
                )
                response = self._call_llm(prompt)
                selected_relations = [r.strip() for r in response.split(",") if r.strip()]
                selected_relations = selected_relations[:self.beam_width]

            entity_relations[entity] = selected_relations
            logger.info(f"实体 '{entity}' 选择的关系: {selected_relations}")

        return entity_relations

    def _explore_entities(
            self,
            entity_relations: Dict[str, List[str]],
            question: str
    ) -> List[Dict[str, Any]]:
        """
        实体探索:为每个(实体,关系)对找到最相关的目标实体
        返回: [{"source": entity, "relation": rel, "targets": [entities]}]
        """
        exploration_results = []

        for source_entity, relations in entity_relations.items():
            for relation in relations:
                # Step 1: Search - 获取所有目标实体
                neighbors = self.neo4j.get_entity_neighbors(source_entity, depth=1)

                # 过滤出使用该关系的邻居
                target_candidates = [
                    n.get("target_entity")
                    for n in neighbors
                    if n.get("relation") == relation and n.get("target_entity")
                ]

                if not target_candidates:
                    continue

                # 去重
                target_candidates = list(dict.fromkeys(target_candidates))

                # Step 2: Prune - 使用LLM选择最相关的实体
                if len(target_candidates) <= self.beam_width:
                    selected_targets = target_candidates
                else:
                    prompt = self.prompts["entity_selection"].format(
                        question=question,
                        relation=relation,
                        entities=", ".join(target_candidates[:20]),  # 限制候选数量
                        beam_width=self.beam_width
                    )
                    response = self._call_llm(prompt)
                    selected_targets = [e.strip() for e in response.split(",") if e.strip()]
                    selected_targets = selected_targets[:self.beam_width]

                exploration_results.append({
                    "source": source_entity,
                    "relation": relation,
                    "targets": selected_targets
                })

                logger.info(f"路径: {source_entity} --[{relation}]--> {selected_targets}")

        return exploration_results

    def _beam_search_iteration(
            self,
            current_paths: List[List[Dict]],
            question: str
    ) -> List[List[Dict]]:
        """
        执行一次beam search迭代
        current_paths: 当前的推理路径列表 [path1, path2, ...]
        每个path是三元组列表: [{"source": e1, "relation": r1, "target": e2}, ...]
        """
        # 获取当前路径的尾实体
        tail_entities = []
        for path in current_paths:
            if path:
                tail_entities.append(path[-1]["target"])
            else:
                # 如果路径为空,从主题实体开始
                continue

        if not tail_entities:
            # 初始化:从主题实体开始
            topic_entities = self._extract_topic_entities(question)
            # 创建初始路径(只包含实体,没有关系)
            current_paths = [[{"source": None, "relation": None, "target": e}] for e in topic_entities]
            tail_entities = topic_entities

        # 1. 关系探索
        entity_relations = self._explore_relations(tail_entities, question)

        # 2. 实体探索
        exploration_results = self._explore_entities(entity_relations, question)

        # 3. 扩展路径
        new_paths = []
        for path in current_paths:
            tail_entity = path[-1]["target"]

            # 找到该尾实体的所有探索结果
            for result in exploration_results:
                if result["source"] == tail_entity:
                    for target in result["targets"]:
                        new_path = path + [{
                            "source": result["source"],
                            "relation": result["relation"],
                            "target": target
                        }]
                        new_paths.append(new_path)

        # 4. 保留top-N路径(beam width)
        # 这里简单返回前N个,可以加入评分机制
        return new_paths[:self.beam_width] if new_paths else current_paths

    # ============================================================
    # Phase 3: Reasoning - 评估和生成答案
    # ============================================================

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

        response = self._call_llm(prompt).lower()
        return "yes" in response

    def _generate_answer(self, question: str, paths: List[List[Dict]]) -> str:
        """基于路径生成答案"""
        paths_text = self._format_paths(paths)
        prompt = self.prompts["answer_generation"].format(
            question=question,
            paths=paths_text
        )

        answer = self._call_llm(prompt, temperature=0.1)
        return answer

    # ============================================================
    # Main Reasoning Entry
    # ============================================================

    def reason(
            self,
            question: str,
            max_depth: Optional[int] = None,
            max_width: Optional[int] = None,
            **kwargs  # 兼容旧接口
    ) -> Dict[str, Any]:
        """
        ToG主推理流程

        Args:
            question: 用户问题
            max_depth: 最大探索深度(可选,覆盖初始化参数)
            max_width: beam宽度(可选,覆盖初始化参数)

        Returns:
            推理结果字典
        """
        start_time = time.time()

        # 使用传入的参数或默认值
        depth_limit = max_depth if max_depth is not None else self.max_depth
        beam_width = max_width if max_width is not None else self.beam_width

        try:
            logger.info(f"开始ToG推理 - 问题: {question}")
            logger.info(f"参数: max_depth={depth_limit}, beam_width={beam_width}")
            if self.neo4j.grag_id:
                logger.info(f"数据隔离: grag_id={self.neo4j.grag_id}")

            # Phase 1: Initialization
            current_paths = []
            reasoning_history = []

            # Phase 2 & 3: Iterative Exploration and Reasoning
            for depth in range(depth_limit):
                logger.info(f"========== 深度 {depth + 1}/{depth_limit} ==========")

                # Exploration
                current_paths = self._beam_search_iteration(current_paths, question)

                if not current_paths:
                    logger.warning(f"深度 {depth + 1}: 无法继续探索")
                    break

                # 记录当前深度的路径
                reasoning_history.append({
                    "depth": depth + 1,
                    "paths": [self._format_paths([p]) for p in current_paths],
                    "num_paths": len(current_paths)
                })

                # Reasoning: 评估是否可以回答
                if self._evaluate_sufficiency(question, current_paths):
                    logger.info(f"深度 {depth + 1}: 信息充足,开始生成答案")
                    break

            # Generate Answer
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