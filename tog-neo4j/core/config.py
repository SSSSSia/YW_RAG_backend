"""
配置管理模块
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""

    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "jbh966225"

    # Java后端回调配置
    java_backend_url: str = "http://localhost:8080"
    java_callback_path: str = "/graph/response"

    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 9090

    # LLM配置
    llm_api_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "qwen3:8b"
    entity_linking_threshold: float = 15.0

    # GraphRAG配置
    graphrag_root: str = "../graphrag"
    base_settings_path: str = "../graphrag/settings.yaml"

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


# 全局配置实例
settings = Settings()