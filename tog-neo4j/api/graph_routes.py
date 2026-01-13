"""
GraphRAG创建图谱接口
"""
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException, status
from pathlib import Path
from models.schemas import R
from services.graph_creation_service import GraphCreationService
from utils.logger import logger
from core.config import settings

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/create", response_model=R)
async def create_graph(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        grag_id: str = Form(...)
):
    """上传文件并创建GraphRAG知识图谱（异步处理）"""
    try:
        logger.info("=" * 60)
        logger.info(f"[{grag_id}] 📊 接收到图谱创建请求")

        # 创建用户目录
        user_path = Path(settings.graphrag_root) / grag_id
        input_dir = user_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{grag_id}] ✅ 目录创建完成: {input_dir}")

        # 保存上传的文件
        file_path = input_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        file_size = file_path.stat().st_size
        logger.info(f"[{grag_id}] ✅ 文件已保存: {file.filename} ({file_size / 1024:.2f} KB)")

        # 创建服务实例并启动后台任务
        service = GraphCreationService(grag_id)
        background_tasks.add_task(
            service.create_graph,
            file_path=str(file_path),
            filename=file.filename
        )

        logger.info(f"[{grag_id}] 📄 后台任务已启动")
        logger.info("=" * 60)

        return R.ok(
            message="正在创建图谱，请稍候...",
            data={
                "status": "processing",
                "grag_id": grag_id,
                "file_saved": file.filename,
                "note": "图谱创建完成后将通过回调接口通知结果"
            }
        )

    except Exception as e:
        logger.error(f"[{grag_id if 'grag_id' in locals() else 'Unknown'}] ❌ 处理失败: {e}", exc_info=True)
        return R.error(
            message="请求处理失败",
            error_detail=str(e),
            code="500"
        )