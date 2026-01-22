"""
FastAPI主入口文件
"""
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import graph_routes, query_routes,agent_routes
from utils.logger import logger
from core.database import db_manager
from services.graph_creation_service import GraphCreationService

# 创建FastAPI应用
app = FastAPI(
    title="ToG Knowledge Graph API",
    description="基于ToG和GraphRAG的知识图谱查询系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(graph_routes.router)
app.include_router(query_routes.router)
app.include_router(agent_routes.router)


# 根路径
@app.get("/", response_model=dict)
async def root():
    return {"message": "ToG Knowledge Graph API is running"}


async def initialize_graph():
    """初始化图数据"""
    try:
        graph_create_service = GraphCreationService(grag_id="2026002_1")
        await graph_create_service.create_graph(
            file_path="../graphrag/2026002_1",
            filename="2026002_1"
        )
        logger.info("图数据初始化完成")
    except Exception as e:
        logger.error(f"图数据初始化失败: {e}")
        raise

if __name__ == "__main__":
    import uvicorn

    # asyncio.run(initialize_graph())

    from core.config import settings

    server_host = settings.server_host
    server_port = settings.server_port

    logger.info("=" * 60)
    logger.info("🚀 启动ToG Knowledge Graph API服务器")
    logger.info(f"📍 地址: http://{server_host}:{server_port}")
    logger.info(f"📚 文档: http://{server_host}:{server_port}/docs")
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host=server_host,
        port=server_port,
        reload=True
    )