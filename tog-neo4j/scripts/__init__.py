# scripts/__init__.py
"""
工具脚本包
包含GraphRAG数据处理、Neo4j导入和索引创建等功能
"""
from .deal_graph import main as extract_graph_data
from .insert_to_neo4j import main as import_to_neo4j
from .ywretriever import crtDenseRetriever as create_dense_retriever, entity_linking

__all__ = [
    "extract_graph_data",
    "import_to_neo4j",
    "create_dense_retriever",
    "entity_linking"
]