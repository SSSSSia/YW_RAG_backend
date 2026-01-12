"""
Neo4j 导入工具
支持多知识库隔离存储，通过 grag_id 区分
"""

import json
import os
from neo4j import GraphDatabase
from typing import List, Dict, Any


class Neo4jImporter:
    """Neo4j 数据导入器，支持知识库隔离"""

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 连接地址
            user: 用户名
            password: 密码
        """
        self.uri = uri
        self.user = user
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[INFO] 已连接到 Neo4j: {uri}")

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("[INFO] Neo4j 连接已关闭")

    def load_json_data(self, json_file: str) -> Dict[str, Any]:
        """加载 JSON 数据文件"""
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"文件未找到: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"[INFO] 加载数据文件: {json_file}")
        print(f"[INFO] 知识库ID: {data['metadata']['grag_id']}")
        print(f"[INFO] 实体数: {data['metadata']['entity_count']}")
        print(f"[INFO] 关系数: {data['metadata']['triple_count']}")

        return data

    def clear_database(self):
        """清空整个数据库（慎用！）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[WARNING] 数据库已清空")

    def clear_knowledge_base(self, grag_id: str):
        """
        删除指定知识库的所有数据（使用 DETACH DELETE 强制删除节点及其所有关系）
        """
        with self.driver.session() as session:
            # 🔴 使用 DETACH DELETE：同时删除节点和它连接的所有关系
            result = session.run("""
                MATCH (n:Entity {grag_id: $grag_id})
                DETACH DELETE n
                RETURN count(n) as deleted_nodes
            """, grag_id=grag_id)

            deleted_nodes = result.single()['deleted_nodes']

        print(f"[INFO] 已删除知识库 '{grag_id}': {deleted_nodes} 个节点及其所有关联关系")

    def create_constraints_and_indexes(self):
        """
        创建约束和索引
        关键：使用 (id, grag_id) 复合约束，确保不同知识库的相同实体独立存储
        """
        with self.driver.session() as session:
            try:
                # 删除旧的单字段约束（如果存在）
                try:
                    session.run("DROP CONSTRAINT entity_id IF EXISTS")
                except:
                    pass

                # 🔴 核心：创建复合唯一约束
                # 相同 id 但不同 grag_id 的实体会被视为不同节点
                session.run("""
                    CREATE CONSTRAINT entity_composite_key IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE (e.id, e.grag_id) IS UNIQUE
                """)
                print("[INFO] ✓ 创建复合唯一约束 (id, grag_id)")

                # 创建索引以加速查询
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                # print("[INFO] ✓ 创建索引: entity_name")

                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                # print("[INFO] ✓ 创建索引: entity_type")

                session.run("CREATE INDEX entity_grag_id IF NOT EXISTS FOR (e:Entity) ON (e.grag_id)")
                # print("[INFO] ✓ 创建索引: entity_grag_id")

            except Exception as e:
                print(f"[WARNING] 创建约束/索引时出现问题: {e}")

    def import_entities(self, entities: Dict[str, Dict[str, Any]]):
        """
        导入实体数据
        使用 (id, grag_id) 联合匹配，确保知识库隔离
        """
        entities_list = []
        for entity_id, entity_data in entities.items():
            entity_data['id'] = entity_id
            entities_list.append(entity_data)

        with self.driver.session() as session:
            # 🔴 核心：MERGE 基于 (id, grag_id) 联合主键
            session.run("""
                UNWIND $entities AS entity
                MERGE (e:Entity {id: entity.id, grag_id: entity.grag_id})
                SET e.name = entity.name,
                    e.type = entity.type,
                    e.description = entity.description,
                    e.degree = entity.degree,
                    e.community_ids = entity.community_ids,
                    e.text_unit_ids = entity.text_unit_ids
            """, entities=entities_list)

        print(f"[INFO] ✓ 导入 {len(entities_list)} 个实体")

    def import_relationships_without_apoc(self, triples: List[Dict[str, Any]]):
        """
        导入关系数据
        关系匹配时同时考虑 grag_id，确保只在同一知识库内建立连接
        """
        if not triples:
            print("[INFO] 无关系需要导入")
            return

        # 自动识别主键字段
        sample = triples[0]
        subject_key = next((k for k in ['subject_id', 'subject', 'source', 'head'] if k in sample), None)
        object_key = next((k for k in ['object_id', 'object', 'target', 'tail'] if k in sample), None)

        if not subject_key or not object_key:
            print("[ERROR] 无法识别主体和客体字段")
            return

        # 判断使用 id 还是 name 匹配
        sample_id_val = str(sample[subject_key])
        is_uuid_like = ('-' in sample_id_val and len(sample_id_val) > 20) or len(sample_id_val) == 32
        match_field = "id" if is_uuid_like else "name"

        print(f"[INFO] 使用字段 '{match_field}' 匹配实体")

        # 按关系类型分组
        relations_by_type = {}
        for triple in triples:
            original_predicate = triple.get('predicate', 'RELATED_TO')
            rel_type = self._normalize_relationship_type(original_predicate)

            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []

            relations_by_type[rel_type].append({
                'subject_val': triple[subject_key],
                'object_val': triple[object_key],
                'weight': triple.get('weight', 0.0),
                'description': triple.get('description', ''),
                'original_predicate': original_predicate,
                'grag_id': triple.get('grag_id', '')  # 🔴 关键：传递 grag_id
            })

        # 批量导入关系
        with self.driver.session() as session:
            for rel_type, rel_triples in relations_by_type.items():
                # 🔴 核心：MATCH 时同时匹配 grag_id，确保只连接同一知识库的节点
                query = f"""
                    UNWIND $triples AS triple
                    MATCH (source:Entity) 
                    WHERE source.{match_field} = triple.subject_val 
                      AND source.grag_id = triple.grag_id
                    MATCH (target:Entity) 
                    WHERE target.{match_field} = triple.object_val 
                      AND target.grag_id = triple.grag_id
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.weight = triple.weight, 
                        r.description = triple.description, 
                        r.original_predicate = triple.original_predicate,
                        r.grag_id = triple.grag_id
                """
                session.run(query, triples=rel_triples)
                # print(f"[INFO] ✓ 导入关系类型 '{rel_type}': {len(rel_triples)} 条")

    def _normalize_relationship_type(self, predicate: str) -> str:
        """标准化关系类型名称"""
        if not predicate or not predicate.strip():
            return 'RELATED_TO'

        normalized = predicate.strip().replace(' ', '_')
        normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized).upper()

        while '__' in normalized:
            normalized = normalized.replace('__', '_')

        normalized = normalized.strip('_')

        if not normalized or normalized.replace('_', '') == '':
            return 'RELATED_TO'

        if normalized[0].isdigit():
            normalized = 'REL_' + normalized

        return normalized

    def get_knowledge_base_stats(self, grag_id: str) -> Dict[str, int]:
        """
        获取指定知识库的统计信息

        Args:
            grag_id: 知识库ID

        Returns:
            统计信息字典
        """
        with self.driver.session() as session:
            # 统计节点数
            result = session.run("""
                MATCH (n:Entity {grag_id: $grag_id})
                RETURN count(n) as node_count
            """, grag_id=grag_id)
            node_count = result.single()['node_count']

            # 统计关系数
            result = session.run("""
                MATCH ()-[r {grag_id: $grag_id}]-()
                RETURN count(r) as rel_count
            """, grag_id=grag_id)
            rel_count = result.single()['rel_count']

        return {
            'grag_id': grag_id,
            'node_count': node_count,
            'relationship_count': rel_count
        }


def main(json_file: str,
         neo4j_uri: str = "bolt://localhost:7687",
         neo4j_user: str = "neo4j",
         neo4j_password: str = "jbh966225",
         clear_existing: bool = False) -> bool:
    """
    主函数：导入数据到 Neo4j

    Args:
        json_file: 由 deal_graph.py 生成的 JSON 文件路径
        neo4j_uri: Neo4j 连接地址
        neo4j_user: Neo4j 用户名
        neo4j_password: Neo4j 密码
        clear_existing: 是否清除已存在的同名知识库

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    importer = None
    try:
        # 连接数据库
        importer = Neo4jImporter(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

        # 加载数据
        data = importer.load_json_data(json_file)
        grag_id = data['metadata']['grag_id']

        # 如果需要，清除已存在的知识库
        if clear_existing:
            importer.clear_knowledge_base(grag_id)

        # 创建约束和索引
        importer.create_constraints_and_indexes()

        # 导入实体
        importer.import_entities(data['entities'])

        # 导入关系
        importer.import_relationships_without_apoc(data['triples'])

        # 显示统计信息
        stats = importer.get_knowledge_base_stats(grag_id)
        print(f"\n{'='*60}")
        print(f"✅ 知识库 '{grag_id}' 导入成功！")
        print(f"{'='*60}")
        print(f"📊 统计信息:")
        print(f"   - 节点数: {stats['node_count']}")
        print(f"   - 关系数: {stats['relationship_count']}")
        print(f"{'='*60}\n")

        return True

    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if importer:
            importer.close()


if __name__ == "__main__":
    from insert_to_neo4j import Neo4jImporter
    # 定义一个修复后的删除方法
    def safe_clear_knowledge_base(importer, grag_id):
        print(f"正在强制清理知识库: {grag_id} ...")
        with importer.driver.session() as session:
            # DETACH DELETE 会自动处理残留的关系
            result = session.run("""
                MATCH (n:Entity {grag_id: $grag_id})
                DETACH DELETE n
                RETURN count(n) as deleted_count
            """, grag_id=grag_id)
            count = result.single()['deleted_count']
        print(f"✅ 成功删除 {count} 个节点及其所有关系。")


    # --- 执行部分 ---
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "jbh966225"
    target_grag_id = "2026001_1"

    importer = Neo4jImporter(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    try:
        safe_clear_knowledge_base(importer, target_grag_id)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        importer.close()
    # # 测试调用
    # test_file = "./output/extracted_data/graph_data.json"
    #
    # success = main(
    #     json_file=test_file,
    #     neo4j_uri="bolt://localhost:7687",
    #     neo4j_user="neo4j",
    #     neo4j_password="jbh966225",
    #     clear_existing=False  # 设为 True 会先删除同名知识库
    # )
    #
    # if success:
    #     print("✅ 流程结束：导入成功")
    # else:
    #     print("❌ 流程结束：导入失败")