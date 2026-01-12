import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
import os
import shutil
import subprocess
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
import sys
from fastapi.middleware.cors import CORSMiddleware

from neo4j_connector import Neo4jConnector
from tog_reasoning import ToGReasoning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVER_PATH = "D:/CODE_FILE/CODE_PYTHON/YW_RAG_backend/tog-neo4j/.retrive/ywcorom"
ENTITY_LINKING_THRESHOLD = 15.0

app = FastAPI(title="ToG Knowledge Graph API")


# ============================================================
# 辅助函数：简洁的subprocess执行
# ============================================================

def run_command_with_progress(command: str, description: str, grag_id: str = None) -> tuple[bool, str, str]:
    """
    执行命令并显示简洁的进度信息

    Args:
        command: 要执行的命令
        description: 操作描述
        grag_id: 用户ID（可选）

    Returns:
        (success, stdout, stderr)
    """
    prefix = f"[{grag_id}]" if grag_id else ""
    logger.info(f"{prefix} 🚀 开始: {description}")
    logger.info(f"{prefix} 💻 命令: {command}")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True
        )

        # 收集输出
        stdout_lines = []
        stderr_lines = []

        # 定义需要显示的关键信息模式
        important_patterns = [
            r'Loading',
            r'Processing',
            r'Creating',
            r'Building',
            r'Indexing',
            r'Complete',
            r'Success',
            r'Error',
            r'Warning',
            r'progress',
            r'⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏',  # 进度spinner
            r'\d+%',  # 百分比
            r'Extracting',
            r'Embedding',
            r'Graph',
        ]

        pattern = re.compile('|'.join(important_patterns), re.IGNORECASE)

        # 实时读取输出
        while True:
            # 读取stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                stdout_lines.append(stdout_line)
                # 只显示重要信息
                clean_line = stdout_line.strip()
                if clean_line and pattern.search(clean_line):
                    # 移除ANSI转义序列
                    clean_line = re.sub(r'\x1B\[[0-9;]*m', '', clean_line)
                    logger.info(f"{prefix} 📝 {clean_line}")

            # 读取stderr
            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_lines.append(stderr_line)
                clean_line = stderr_line.strip()
                if clean_line and pattern.search(clean_line):
                    clean_line = re.sub(r'\x1B\[[0-9;]*m', '', clean_line)
                    logger.warning(f"{prefix} ⚠️  {clean_line}")

            # 检查进程是否结束
            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                break

        returncode = process.wait()
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        if returncode == 0:
            logger.info(f"{prefix} ✅ 完成: {description}")
            return True, stdout, stderr
        else:
            logger.error(f"{prefix} ❌ 失败: {description} (返回码: {returncode})")
            return False, stdout, stderr

    except Exception as e:
        logger.error(f"{prefix} ❌ 异常: {description} - {str(e)}")
        return False, "", str(e)


def log_step(step_num: int, total_steps: int, description: str, grag_id: str = None):
    """记录步骤信息"""
    prefix = f"[{grag_id}]" if grag_id else ""
    logger.info(f"{prefix} 📍 步骤 {step_num}/{total_steps}: {description}")


# ============================================================
# 配置CORS - 允许跨域请求
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 数据库连接管理
# ============================================================

db_connections: Dict[str, Neo4jConnector] = {}

DEFAULT_NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "jbh966225"
}


def get_neo4j_connector(kg_name: Optional[str] = None) -> Neo4jConnector:
    """获取或创建 Neo4j 连接实例"""
    if not kg_name:
        kg_name = "default"

    if kg_name in db_connections:
        return db_connections[kg_name]

    try:
        connector = Neo4jConnector(
            uri=DEFAULT_NEO4J_CONFIG["uri"],
            username=DEFAULT_NEO4J_CONFIG["username"],
            password=DEFAULT_NEO4J_CONFIG["password"]
        )
        db_connections[kg_name] = connector
        logger.info(f"为数据库 '{kg_name}' 创建新连接")
        return connector
    except Exception as e:
        logger.error(f"创建数据库连接失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"无法连接到数据库 '{kg_name}': {str(e)}"
        )


@app.get("/CORS_test")
async def index():
    """简单的测试接口，用于验证跨域(CORS)配置是否生效"""
    logger.info("收到 CORS跨域 测试请求")
    return {
        "message": "CORS test successful",
        "status": "ok"
    }


# ============================================================
# 请求和响应模型
# ============================================================

class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: str


class ToGQueryRequest(BaseModel):
    """ToG查询请求"""
    kg_name: Optional[str] = None
    max_depth: Optional[int] = 10
    max_width: Optional[int] = 3
    messages: Optional[List[MessageItem]] = None


class ToGQueryResponse(BaseModel):
    """ToG查询响应"""
    success: bool
    question: str
    answer: str
    execution_time: float
    error_message: Optional[str] = None


class GraphRAGQueryRequest(BaseModel):
    """GraphRAG查询请求"""
    grag_id: str
    messages: Optional[List[MessageItem]] = None
    method: Optional[str] = "local"


class GraphRAGQueryResponse(BaseModel):
    """GraphRAG查询响应"""
    success: bool
    question: str
    answer: str
    grag_id: str
    execution_time: Optional[float] = 0
    error_message: Optional[str] = None


# ============================================================
# ToG查询端点
# ============================================================

@app.post("/TOG_graph", response_model=ToGQueryResponse)
async def query_with_tog(request: ToGQueryRequest):
    """使用ToG (Think-on-Graph) 方法查询知识图谱"""
    try:
        logger.info("=" * 60)
        logger.info("🔍 收到ToG查询请求")

        # 1. 解析 Message
        question = None
        conversation_history = []

        if request.messages and len(request.messages) > 0:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            for message in reversed(request.messages):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"❌ {error_msg}")
            return ToGQueryResponse(
                success=False,
                question="",
                answer="",
                execution_time=0,
                error_message=error_msg
            )

        logger.info(f"💬 问题: {question}")
        logger.info(f"🗄️  数据库: {request.kg_name or 'default'}")

        # 2. 获取数据库连接
        log_step(1, 3, "连接数据库")
        neo4j_connector = get_neo4j_connector(request.kg_name)
        logger.info("✅ 数据库连接成功")

        # 3. 创建 ToG 推理引擎
        log_step(2, 3, "初始化ToG推理引擎")
        tog_reasoning = ToGReasoning(
            neo4j_connector=neo4j_connector,
            llm_model="qwen3:8b",
            api_key="",
            beam_width=request.max_width or 3,
            max_depth=request.max_depth or 10,
            retriever_path=RETRIEVER_PATH,
            entity_linking_threshold=ENTITY_LINKING_THRESHOLD
        )
        logger.info("✅ ToG引擎初始化完成")

        # 4. 执行ToG推理
        log_step(3, 3, "执行ToG推理")
        result = tog_reasoning.reason(
            question=question,
            max_depth=request.max_depth or 10,
            max_width=request.max_width or 3
        )

        logger.info(f"✅ 查询完成，耗时: {result.get('execution_time', 0):.2f}秒")
        logger.info(f"📄 答案长度: {len(result.get('answer', ''))} 字符")
        logger.info("=" * 60)

        return ToGQueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 查询处理失败: {e}", exc_info=True)
        return ToGQueryResponse(
            success=False,
            question=question if 'question' in locals() else "",
            answer="",
            execution_time=0,
            error_message=f"查询处理失败: {str(e)}"
        )


# ============================================================
# GraphRAG 相关端点
# ============================================================

GRAPHRAG_ROOT = "../graphrag"
BASE_SETTINGS_PATH = os.path.join(GRAPHRAG_ROOT, "settings.yaml")


@app.post("/create_graph")
async def creat_graph(
        file: UploadFile = File(...),
        grag_id: str = Form(...)
):
    """
    上传文件并创建GraphRAG知识图谱
    """
    try:
        logger.info("=" * 60)
        logger.info(f"📊 开始创建GraphRAG知识图谱 - 用户: {grag_id}")

        TOTAL_STEPS = 5

        # 步骤1: 创建用户目录
        log_step(1, TOTAL_STEPS, "创建用户目录", grag_id)
        user_path = os.path.join(GRAPHRAG_ROOT, grag_id)
        input_dir = os.path.join(user_path, "input")
        os.makedirs(input_dir, exist_ok=True)
        logger.info(f"[{grag_id}] ✅ 目录创建完成: {input_dir}")

        # 步骤2: 保存上传的文件
        log_step(2, TOTAL_STEPS, f"保存文件: {file.filename}", grag_id)
        file_path = os.path.join(input_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)
        logger.info(f"[{grag_id}] ✅ 文件已保存: {file.filename} ({file_size / 1024:.2f} KB)")

        # 步骤3: 初始化GraphRAG
        log_step(3, TOTAL_STEPS, "初始化GraphRAG配置", grag_id)
        init_command = f"python -m graphrag init --root {user_path}"

        success, stdout, stderr = run_command_with_progress(
            init_command,
            "GraphRAG初始化",
            grag_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"初始化失败: {stderr}"
            )

        # 步骤4: 复制配置文件
        log_step(4, TOTAL_STEPS, "配置settings.yaml", grag_id)
        user_settings_path = os.path.join(user_path, "settings.yaml")
        if os.path.exists(BASE_SETTINGS_PATH):
            shutil.copy2(BASE_SETTINGS_PATH, user_settings_path)
            logger.info(f"[{grag_id}] ✅ 配置文件已复制")
        else:
            logger.warning(f"[{grag_id}] ⚠️  基础配置文件不存在: {BASE_SETTINGS_PATH}")

        # 步骤5: 构建索引
        log_step(5, TOTAL_STEPS, "构建知识图谱索引 (这可能需要几分钟)", grag_id)
        index_command = f"python -m graphrag index --root {user_path}"

        success, stdout, stderr = run_command_with_progress(
            index_command,
            "索引构建",
            grag_id
        )

        if not success:
            logger.error(f"[{grag_id}] ❌ 索引构建失败")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": "索引构建失败",
                    "error": stderr[:500],  # 只返回前500字符
                    "user_directory": grag_id,
                    "file_saved": file.filename
                }
            )

        logger.info(f"[{grag_id}] 🎉 GraphRAG知识图谱创建成功！")
        logger.info("=" * 60)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "处理完成",
                "user_directory": grag_id,
                "file_saved": file.filename,
                "output_path": os.path.join(user_path, "output")
            }
        )

    except Exception as e:
        logger.error(f"[{grag_id if 'grag_id' in locals() else 'Unknown'}] ❌ 处理失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "处理失败",
                "error": str(e),
                "user_directory": grag_id if 'grag_id' in locals() else None,
                "file_saved": file.filename if 'file' in locals() and file else None
            }
        )


def clean_graphrag_output(raw_text: str) -> str:
    """清理GraphRAG的原始输出"""
    # 去除ANSI转义序列
    ansi_escape = re.compile(r'(\x1B\[[0-9;]*m|\[[0-9;]*m)')
    text = ansi_escape.sub('', raw_text)

    # 去除引用标记
    text = re.sub(r'\[Data: [^\]]+\]', '', text)

    # 清理空白
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text


@app.post("/graphrag_query", response_model=GraphRAGQueryResponse)
async def graphrag_query(request: GraphRAGQueryRequest):
    """执行GraphRAG查询"""
    import time
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info(f"[{request.grag_id}] 🔍 收到GraphRAG查询请求")

        # 1. 解析 messages
        question = None
        conversation_history = []

        if request.messages and len(request.messages) > 0:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            for message in reversed(request.messages):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return GraphRAGQueryResponse(
                success=False,
                question="",
                answer="",
                grag_id=request.grag_id,
                execution_time=0,
                error_message=error_msg
            )

        logger.info(f"[{request.grag_id}] 💬 问题: {question}")
        logger.info(f"[{request.grag_id}] 🔧 方法: {request.method}")

        # 2. 检查用户目录
        log_step(1, 2, "检查知识图谱目录", request.grag_id)
        user_path = os.path.join(GRAPHRAG_ROOT, request.grag_id)
        if not os.path.exists(user_path):
            error_msg = f"目录 {request.grag_id} 不存在，请先创建知识图谱"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return GraphRAGQueryResponse(
                success=False,
                question=question,
                answer="",
                grag_id=request.grag_id,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

        logger.info(f"[{request.grag_id}] ✅ 知识图谱目录存在")

        # 3. 执行查询
        log_step(2, 2, "执行GraphRAG查询", request.grag_id)
        query_command = (
            f'python -m graphrag query '
            f'--root {user_path} '
            f'--method {request.method} '
            f'--query "{question}"'
        )

        success, stdout, stderr = run_command_with_progress(
            query_command,
            f"GraphRAG {request.method} 查询",
            request.grag_id
        )

        execution_time = time.time() - start_time

        if success:
            result = stdout.strip()

            logger.info(f"[{request.grag_id}] ✅ 查询成功，耗时: {execution_time:.2f}秒")
            logger.info(f"[{request.grag_id}] 📄 答案长度: {len(result)} 字符")
            logger.info("=" * 60)

            return GraphRAGQueryResponse(
                success=True,
                question=question,
                answer=result,
                grag_id=request.grag_id,
                execution_time=execution_time,
                error_message=None
            )
        else:
            error_msg = stderr[:500] if stderr else "未知错误"
            logger.error(f"[{request.grag_id}] ❌ 查询失败: {error_msg}")
            logger.info("=" * 60)
            return GraphRAGQueryResponse(
                success=False,
                question=question,
                answer="",
                grag_id=request.grag_id,
                execution_time=execution_time,
                error_message=f"查询失败: {error_msg}"
            )

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        error_msg = "查询执行超时(超过5分钟)"
        logger.error(f"[{request.grag_id}] ❌ {error_msg}")
        logger.info("=" * 60)
        return GraphRAGQueryResponse(
            success=False,
            question=question if 'question' in locals() else "",
            answer="",
            grag_id=request.grag_id,
            execution_time=execution_time,
            error_message=error_msg
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[{request.grag_id}] ❌ 查询异常: {e}", exc_info=True)
        logger.info("=" * 60)
        return GraphRAGQueryResponse(
            success=False,
            question=question if 'question' in locals() else "",
            answer="",
            grag_id=request.grag_id,
            execution_time=execution_time,
            error_message=f"查询处理失败: {str(e)}"
        )


# ============================================================
# 启动服务器
# ============================================================

if __name__ == "__main__":
    import uvicorn

    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "9090"))

    logger.info("=" * 60)
    logger.info("🚀 启动ToG Knowledge Graph API服务器")
    logger.info(f"📍 地址: http://{server_host}:{server_port}")
    logger.info(f"📚 文档: http://{server_host}:{server_port}/docs")
    logger.info("=" * 60)

    uvicorn.run(
        "fastapi_server:app",
        host=server_host,
        port=server_port,
        reload=True
    )