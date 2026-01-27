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
    java_callback_path: str = "/graphs/result"
    java_get_knowledge_bases_path: str = "/graphs/list"

    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 9090

    # LLM配置 (Ollama)
    llm_api_url: str = "http://localhost:11434"  # Ollama 默认服务地址，不需要额外的API路径
    llm_model: str = "qwen3:8b"
    entity_linking_threshold: float = 15.0
    llm_timeout: int = 120
    llm_max_retries: int = 3

    # 硅基流动API配置
    siliconflow_api_url: str
    siliconflow_api_key: str
    siliconflow_model: str
    siliconflow_vision_model: str
    siliconflow_timeout: int = 120
    siliconflow_max_retries: int = 3

    # GraphRAG配置
    graphrag_root: str = "../graphrag"
    base_settings_path: str = "../graphrag/settings.yaml"

    # MySQL配置
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_table: str
    mysql_charset: str

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


# 全局配置实例
settings = Settings()