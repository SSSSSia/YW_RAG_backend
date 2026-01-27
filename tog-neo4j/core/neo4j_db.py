"""
Neo4j数据库连接管理
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from utils.logger import logger
from core.config import settings


class Neo4jConnector:
    """Neo4j数据库连接器 - 支持grag_id隔离"""

    def __init__(self, uri: str, username: str, password: str, grag_id: Optional[str] = None):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self.grag_id = grag_id
            logger.info(f"成功连接到Neo4j数据库 (grag_id: {grag_id})")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def execute_query(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
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
        """为查询添加grag_id过滤条件"""
        if not self.grag_id:
            return query

        connector = "AND" if where_exists else "WHERE"
        import re
        node_vars = re.findall(r'\((\w+)(?::\w+)?(?:\s*\{[^}]*\})?\)', query)

        if not node_vars:
            return query

        filters = [f"{var}.grag_id = $grag_id" for var in set(node_vars)]
        filter_clause = f" {connector} " + " AND ".join(filters)

        if "RETURN" in query:
            parts = query.split("RETURN", 1)
            return parts[0] + filter_clause + " RETURN" + parts[1]
        elif "WITH" in query:
            parts = query.split("WITH", 1)
            return parts[0] + filter_clause + " WITH" + parts[1]
        else:
            return query + filter_clause

    def _get_params_with_grag_id(self, params: Dict = None) -> Dict:
        """为参数字典添加grag_id"""
        params = params or {}
        if self.grag_id:
            params["grag_id"] = self.grag_id
        return params

    def search_entities_exact(self, entity_name: str) -> List[Dict]:
        """精确匹配实体"""
        query = """
        MATCH (n)
        WHERE n.name = $entity_name AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT 10
        """
        params = self._get_params_with_grag_id({"entity_name": entity_name})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_fuzzy(self, keyword: str, limit: int = 10) -> List[Dict]:
        """模糊匹配实体"""
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($keyword) AND n.grag_id = $grag_id
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = self._get_params_with_grag_id({"keyword": keyword, "limit": limit})
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> List[Dict]:
        """获取实体的邻居节点"""
        query = f"""
        MATCH path = (n)-[r*1..{depth}]-(m)
        WHERE n.name = $entity_name
          AND n.grag_id = $grag_id
          AND m.grag_id = $grag_id
          AND all(rel in r WHERE rel.grag_id = $grag_id)
        RETURN DISTINCT
            n.name as source_entity,
            type(r[0]) as relation,
            m.name as target_entity
        LIMIT 100
        """
        params = self._get_params_with_grag_id({"entity_name": entity_name})
        return self.execute_query(query, params)

    def get_relation_path(self, source: str, target: str, max_depth: int = 3) -> List[Dict]:
        """查找两个实体之间的关系路径"""
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


class DatabaseManager:
    """数据库连接管理器"""

    def __init__(self):
        self.connections: Dict[str, Neo4jConnector] = {}

    def get_connector(self, grag_id: str) -> Neo4jConnector:
        """获取或创建Neo4j连接实例"""
        cache_key = f"connector_{grag_id}"

        if cache_key in self.connections:
            return self.connections[cache_key]

        connector = Neo4jConnector(
            uri=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password,
            grag_id=grag_id
        )
        self.connections[cache_key] = connector
        logger.info(f"为图谱 '{grag_id}' 创建新连接")
        return connector

    def close_all(self):
        """关闭所有连接"""
        for connector in self.connections.values():
            connector.close()
        self.connections.clear()


# 全局数据库管理器实例
db_manager = DatabaseManager()


# ==================== 运维流程独立的Neo4j访问类 ====================

class YWOperationNeo4j:
    """
    运维操作流程专用Neo4j访问类
    与GraphRAG逻辑完全独立，不使用grag_id隔离
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        初始化运维流程Neo4j连接

        Args:
            uri: Neo4j连接地址，默认从配置读取
            username: 用户名，默认从配置读取
            password: 密码，默认从配置读取
        """
        from core.config import settings

        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_user
        self.password = password or settings.neo4j_password

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info(f"成功连接到运维流程Neo4j数据库: {self.uri}")
        except Exception as e:
            logger.error(f"连接运维流程Neo4j失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("运维流程Neo4j连接已关闭")

    def execute_query(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"运维流程Cypher查询失败: {e}")
            logger.error(f"查询语句: {cypher_query}")
            logger.error(f"参数: {parameters}")
            return []

    def get_all_operation_processes(self) -> List[str]:
        """
        获取所有操作流程名称（operation_process属性值）

        Returns:
            操作流程名称列表
        """
        query = """
        MATCH (n)
        WHERE n.operation_process IS NOT NULL
        RETURN DISTINCT n.operation_process AS process_name
        ORDER BY process_name
        """
        results = self.execute_query(query)
        return [str(item["process_name"]) for item in results if item.get("process_name")]

    def get_operation_process_nodes(self, process_name: str) -> List[str]:
        """
        获取指定操作流程的所有节点名称

        Args:
            process_name: 操作流程名称（operation_process属性值）

        Returns:
            该流程中所有节点的名称列表
        """
        query = """
        MATCH (n)
        WHERE n.operation_process = $process_name
        RETURN DISTINCT n.name AS node_name
        ORDER BY node_name
        """
        results = self.execute_query({"process_name": process_name})
        return [str(item["node_name"]) for item in results if item.get("node_name")]

    def get_operation_process_chain(self, process_name: str) -> List[Dict[str, Any]]:
        """
        获取指定操作流程的完整链条（带关系信息）

        Args:
            process_name: 操作流程名称

        Returns:
            包含节点和关系信息的链条列表
            格式: [{"from": "节点A", "to": "节点B", "relation": "下一步"}, ...]
        """
        query = """
        MATCH (n1)-[r:下一步]->(n2)
        WHERE n1.operation_process = $process_name
          AND n2.operation_process = $process_name
        RETURN DISTINCT
            n1.name AS from_node,
            n2.name AS to_node,
            type(r) AS relation
        ORDER BY from_node, to_node
        """
        results = self.execute_query({"process_name": process_name})

        return [
            {
                "from": str(item["from_node"]),
                "to": str(item["to_node"]),
                "relation": str(item["relation"])
            }
            for item in results
            if item.get("from_node") and item.get("to_node")
        ]


# 全局运维流程Neo4j实例（延迟初始化）
_yw_neo4j_instance: Optional[YWOperationNeo4j] = None


def get_yw_neo4j() -> YWOperationNeo4j:
    """获取运维流程Neo4j单例"""
    global _yw_neo4j_instance
    if _yw_neo4j_instance is None:
        _yw_neo4j_instance = YWOperationNeo4j()
    return _yw_neo4j_instance
