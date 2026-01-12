"""
Neo4j 数据库连接模块-（支持 grag_id 过滤）
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Neo4j 数据库连接器 - 支持 grag_id 隔离"""

    def __init__(self, uri: str, username: str, password: str, grag_id: Optional[str] = None):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 数据库地址,如 "bolt://localhost:7687"
            username: 用户名
            password: 密码
            grag_id: 图谱ID，用于数据隔离
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self.grag_id = grag_id
            logger.info(f"成功连接到 Neo4j 数据库 (grag_id: {grag_id})")
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")

    def execute_query(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """
        执行 Cypher 查询

        Args:
            cypher_query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Cypher查询失败: {e}")
            logger.error(f"查询语句: {cypher_query}")
            logger.error(f"参数: {parameters}")
            return []

    def _add_grag_filter(self, query: str, where_exists: bool = True) -> str:
        """
        为查询添加 grag_id 过滤条件

        Args:
            query: 原始查询语句
            where_exists: 查询中是否已存在 WHERE 子句

        Returns:
            添加了 grag_id 过滤的查询语句
        """
        if not self.grag_id:
            return query

        # 如果已有 WHERE，使用 AND；否则添加 WHERE
        connector = "AND" if where_exists else "WHERE"

        # 查找所有节点变量（简单实现，匹配 MATCH 后的括号中的变量）
        import re
        node_vars = re.findall(r'\((\w+)(?::\w+)?(?:\s*\{[^}]*\})?\)', query)

        if not node_vars:
            return query

        # 为每个节点变量添加 grag_id 过滤
        filters = [f"{var}.grag_id = $grag_id" for var in set(node_vars)]
        filter_clause = f" {connector} " + " AND ".join(filters)

        # 在 RETURN 前插入过滤条件
        if "RETURN" in query:
            parts = query.split("RETURN", 1)
            return parts[0] + filter_clause + " RETURN" + parts[1]
        elif "WITH" in query:
            parts = query.split("WITH", 1)
            return parts[0] + filter_clause + " WITH" + parts[1]
        else:
            return query + filter_clause

    def _get_params_with_grag_id(self, params: Dict = None) -> Dict:
        """为参数字典添加 grag_id"""
        params = params or {}
        if self.grag_id:
            params["grag_id"] = self.grag_id
        return params

    def search_entities_exact(self, entity_name: str) -> List[Dict]:
        """
        精确匹配实体（限制在当前 grag_id）
        """
        query = """
        MATCH (n)
        WHERE n.name = $entity_name AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT 10
        """
        params = self._get_params_with_grag_id({"entity_name": entity_name})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_partial(self, keyword: str) -> List[Dict]:
        """
        部分匹配实体（包含关键词，限制在当前 grag_id）
        """
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT 10
        """
        params = self._get_params_with_grag_id({"keyword": keyword})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_fuzzy(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        模糊匹配实体（限制在当前 grag_id）
        """
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($keyword) AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = self._get_params_with_grag_id({"keyword": keyword, "limit": limit})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_containing(self, keyword: str, limit: int = 5) -> List[Dict]:
        """
        搜索包含特定关键词的实体（限制在当前 grag_id）
        """
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = self._get_params_with_grag_id({"keyword": keyword, "limit": limit})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        搜索实体 - 支持多种匹配策略（限制在当前 grag_id）
        """
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL 
          AND toLower(n.name) CONTAINS toLower($keyword)
          AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = self._get_params_with_grag_id({"keyword": keyword, "limit": limit})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> List[Dict]:
        """
        获取实体的邻居节点（限制在当前 grag_id）

        Args:
            entity_name: 实体名称
            depth: 遍历深度

        Returns:
            邻居节点列表
        """
        query = f"""
        MATCH path = (n)-[r*1..{depth}]-(m)
        WHERE n.name = $entity_name 
          AND n.grag_id = $grag_id 
          AND m.grag_id = $grag_id
          AND all(rel in r WHERE rel.grag_id = $grag_id)
        RETURN DISTINCT 
            n.name as source_entity,
            type(r[0]) as relation,
            m.name as target_entity,
            labels(m) as target_labels,
            properties(m) as target_properties
        LIMIT 100
        """
        params = self._get_params_with_grag_id({"entity_name": entity_name})
        return self.execute_query(query, params)

    def get_relation_path(self, source: str, target: str, max_depth: int = 3) -> List[Dict]:
        """
        查找两个实体之间的关系路径（限制在当前 grag_id）

        Args:
            source: 源实体名称
            target: 目标实体名称
            max_depth: 最大路径长度

        Returns:
            路径列表
        """
        query = f"""
        MATCH path = shortestPath(
            (s)-[*1..{max_depth}]-(t)
        )
        WHERE s.name = $source 
          AND t.name = $target
          AND s.grag_id = $grag_id
          AND t.grag_id = $grag_id
          AND all(rel in relationships(path) WHERE rel.grag_id = $grag_id)
        RETURN 
            [node in nodes(path) | node.name] as node_names,
            [rel in relationships(path) | type(rel)] as relation_types,
            length(path) as path_length
        LIMIT 5
        """
        params = self._get_params_with_grag_id({"source": source, "target": target})
        return self.execute_query(query, params)

    def get_subgraph(self, entity_names: List[str], depth: int = 2) -> Dict[str, Any]:
        """
        获取多个实体周围的子图（限制在当前 grag_id）

        Args:
            entity_names: 实体名称列表
            depth: 子图深度

        Returns:
            子图数据(节点和边)
        """
        query = f"""
        MATCH path = (n)-[r*1..{depth}]-(m)
        WHERE n.name IN $entity_names
          AND n.grag_id = $grag_id
          AND m.grag_id = $grag_id
          AND all(rel in r WHERE rel.grag_id = $grag_id)
        UNWIND nodes(path) as node
        UNWIND relationships(path) as rel
        WITH DISTINCT node, rel
        RETURN 
            collect(DISTINCT {{
                id: id(node),
                name: node.name,
                labels: labels(node),
                properties: properties(node)
            }}) as nodes,
            collect(DISTINCT {{
                source: id(startNode(rel)),
                target: id(endNode(rel)),
                type: type(rel),
                properties: properties(rel)
            }}) as relationships
        """
        params = self._get_params_with_grag_id({"entity_names": entity_names})
        result = self.execute_query(query, params)
        return result[0] if result else {"nodes": [], "relationships": []}

    def execute_complex_query(self, cypher_query: str) -> List[Dict]:
        """
        执行复杂的自定义 Cypher 查询
        用于 LLM 生成的查询语句

        注意：会自动添加 grag_id 过滤

        Args:
            cypher_query: 完整的 Cypher 查询语句

        Returns:
            查询结果
        """
        try:
            # 自动添加 grag_id 过滤
            filtered_query = self._add_grag_filter(cypher_query, where_exists="WHERE" in cypher_query)
            params = self._get_params_with_grag_id()
            return self.execute_query(filtered_query, params)
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            logger.error(f"查询语句: {cypher_query}")
            return []

    def search_operations_by_keyword(self, keyword: str) -> List[Dict]:
        """
        专门搜索操作节点（Operation）（限制在当前 grag_id）
        用于处理 "系统激活" 这类查询
        """
        query = """
        MATCH (op:Operation)
        WHERE toLower(op.name) CONTAINS toLower($keyword)
          AND op.grag_id = $grag_id
        RETURN op.name AS entity_name
        LIMIT 10
        """
        params = self._get_params_with_grag_id({"keyword": keyword})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def get_operation_steps(self, operation_name: str) -> List[Dict]:
        """
        获取某个操作的所有步骤（限制在当前 grag_id）
        """
        query = """
        MATCH (op:Operation {name: $operation_name})-[:HAS_STEP]->(step:Step)
        WHERE op.grag_id = $grag_id 
          AND step.grag_id = $grag_id
        OPTIONAL MATCH path = (step)-[:NEXT_STEP*0..]->(nextStep:Step)
        WHERE all(s in nodes(path) WHERE s.grag_id = $grag_id)
          AND all(rel in relationships(path) WHERE rel.grag_id = $grag_id)
        WITH step, path
        ORDER BY length(path)
        RETURN DISTINCT step.name AS step_name
        """
        params = self._get_params_with_grag_id({"operation_name": operation_name})
        return self.execute_query(query, params)

    def get_operation_flow(self, operation_name: str) -> Dict[str, Any]:
        """
        获取操作的完整流程（包含步骤顺序）（限制在当前 grag_id）
        """
        query = """
        MATCH (op:Operation {name: $operation_name})-[:HAS_STEP]->(firstStep:Step)
        WHERE op.grag_id = $grag_id
          AND firstStep.grag_id = $grag_id
          AND NOT (:Step)-[:NEXT_STEP]->(firstStep)
        MATCH path = (firstStep)-[:NEXT_STEP*0..]->(step:Step)
        WHERE all(s in nodes(path) WHERE s.grag_id = $grag_id)
          AND all(rel in relationships(path) WHERE rel.grag_id = $grag_id)
        WITH nodes(path) as steps
        RETURN [s in steps | s.name] as flow
        """
        params = self._get_params_with_grag_id({"operation_name": operation_name})
        result = self.execute_query(query, params)
        if result and result[0].get("flow"):
            return {
                "operation": operation_name,
                "steps": result[0]["flow"]
            }
        return {"operation": operation_name, "steps": []}