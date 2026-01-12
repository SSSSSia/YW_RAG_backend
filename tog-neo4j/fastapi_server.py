import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import shutil
import subprocess
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime
from neo4j_connector import Neo4jConnector
from tog_reasoning import ToGReasoning
from deal_graph import main as deal_graph_main
from insert_to_neo4j import main as insert_neo4j_main
from ywretriever import crtDenseRetriever


# ====================================================================================================================================================================================
# 配置信息
# ====================================================================================================================================================================================


# ============================================================
# 配置CORS - 允许跨域请求
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVER_PATH_BASE = "../graphrag"
ENTITY_LINKING_THRESHOLD = 15.0

# Java后端接口配置
JAVA_BACKEND_URL = os.getenv("JAVA_BACKEND_URL", "http://localhost:8080")  # 根据实际情况修改
JAVA_CALLBACK_PATH = "/graph/response"

app = FastAPI(title="ToG Knowledge Graph API")

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

GRAPHRAG_ROOT = "../graphrag"
BASE_SETTINGS_PATH = os.path.join(GRAPHRAG_ROOT, "settings.yaml")

# ====================================================================================================================================================================================
# /配置信息
# ====================================================================================================================================================================================



# ====================================================================================================================================================================================
# 请求和响应模型
# ====================================================================================================================================================================================


class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: str


class ToGQueryRequest(BaseModel):
    """ToG查询请求（修改：添加 grag_id）"""
    grag_id: str  # 必需参数
    max_depth: Optional[int] = 10
    max_width: Optional[int] = 3
    messages: Optional[List[MessageItem]] = None


class ToGQueryResponse(BaseModel):
    """ToG查询响应"""
    success: bool
    question: str
    answer: str
    execution_time: float
    grag_id: str  # 添加 grag_id 到响应
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


class ToGGraphRAGQueryRequest(BaseModel):
    """ToG+GraphRAG混合查询请求"""
    grag_id: str  # 必需参数
    max_depth: Optional[int] = 10  # ToG参数
    max_width: Optional[int] = 3   # ToG参数
    method: Optional[str] = "local"  # GraphRAG参数
    messages: Optional[List[MessageItem]] = None


class ToGGraphRAGQueryResponse(BaseModel):
    """ToG+GraphRAG混合查询响应"""
    success: bool
    question: str
    final_answer: str  # 整合后的最终答案
    tog_answer: str    # ToG原始答案
    graphrag_answer: str  # GraphRAG原始答案
    grag_id: str
    execution_time: float
    error_message: Optional[str] = None

# ====================================================================================================================================================================================
# /请求和响应模型
# ====================================================================================================================================================================================


# ====================================================================================================================================================================================
# 工具函数
# ====================================================================================================================================================================================

# ============================================================
# 回调通知函数
# ============================================================

async def notify_java_backend(grag_id: str, success: bool, message: str,
                              file_saved: Optional[str] = None,
                              error: Optional[str] = None,
                              output_path: Optional[str] = None,
                              json_extracted: Optional[str] = None):
    """
    通知Java后端图谱创建结果

    Args:
        grag_id: 用户ID
        success: 是否成功
        message: 结果消息
        file_saved: 保存的文件名
        error: 错误信息
        output_path: 输出路径
        json_extracted: 提取的JSON文件路径
    """
    callback_url = f"{JAVA_BACKEND_URL}{JAVA_CALLBACK_PATH}"

    payload = {
        "grag_id": grag_id,
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "file_saved": file_saved,
        "error": error,
        "output_path": output_path,
        "json_extracted": json_extracted,
        "database_imported": success  # 成功时表示已导入数据库
    }

    try:
        logger.info(f"[{grag_id}] 📤 发送结果通知到Java后端: {callback_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)

            if response.status_code == 200:
                logger.info(f"[{grag_id}] ✅ 成功通知Java后端")
            else:
                logger.warning(f"[{grag_id}] ⚠️ Java后端返回非200状态码: {response.status_code}")

    except httpx.TimeoutException:
        logger.error(f"[{grag_id}] ❌ 通知Java后端超时")
    except Exception as e:
        logger.error(f"[{grag_id}] ❌ 通知Java后端失败: {e}", exc_info=True)


# ============================================================
# 导出节点到CSV函数
# ============================================================

def export_nodes_to_csv(grag_id: str, user_path: str) -> bool:
    """
    从Neo4j导出节点数据到CSV文件

    Args:
        grag_id: 图谱ID
        user_path: 用户目录路径

    Returns:
        是否成功
    """
    try:
        logger.info(f"[{grag_id}] 📤 开始导出节点到CSV")

        # 使用已有的 Neo4j 连接配置
        connector = get_neo4j_connector(grag_id)

        # 修改查询：添加 WHERE n.grag_id = $grag_id 条件
        query = """
        MATCH (n)
        WHERE n.grag_id = $grag_id
        RETURN elementId(n) AS id, COALESCE(n.name, '') AS name
        """

        with connector.driver.session() as session:
            # 在 session.run 中传递参数 grag_id
            result = session.run(query, {"grag_id": grag_id})

            # 收集结果数据
            nodes_data = [record.data() for record in result]

            if not nodes_data:
                logger.warning(f"[{grag_id}] ⚠️ 数据库中没有匹配该 grag_id 的节点数据")
                return False

            # 转换为DataFrame并保存
            import pandas as pd
            df = pd.DataFrame(nodes_data)

            # 保存到用户目录中，便于后续处理
            csv_path = os.path.join(user_path, "nodes_pandas.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')

            logger.info(f"[{grag_id}] ✅ 节点导出完成: {csv_path} ({len(nodes_data)} 个节点)")
            return True


    except Exception as e:
        logger.error(f"[{grag_id}] ❌ 导出节点到CSV失败: {e}", exc_info=True)
        return False


# ============================================================
# 后台任务：异步创建图谱
# ============================================================

async def create_graph_task(file_path: str, filename: str, grag_id: str,
                            user_path: str, input_dir: str):
    """
    后台任务：执行图谱创建的完整流程

    Args:
        file_path: 上传文件的完整路径
        filename: 文件名
        grag_id: 用户ID
        user_path: 用户目录路径
        input_dir: 输入目录路径
    """
    try:
        logger.info(f"[{grag_id}] 📄 开始后台图谱创建任务")
        TOTAL_STEPS = 7

        # 步骤1: 初始化GraphRAG
        log_step(1, TOTAL_STEPS, "初始化GraphRAG配置", grag_id)
        init_command = f"python -m graphrag init --root {user_path}"

        success, stdout, stderr = run_command_with_progress(
            init_command,
            "GraphRAG初始化",
            grag_id
        )

        if not success:
            await notify_java_backend(
                grag_id=grag_id,
                success=False,
                message="初始化失败",
                file_saved=filename,
                error=stderr[:500]
            )
            return

        # 步骤2: 复制配置文件
        log_step(2, TOTAL_STEPS, "配置settings.yaml", grag_id)
        user_settings_path = os.path.join(user_path, "settings.yaml")
        if os.path.exists(BASE_SETTINGS_PATH):
            shutil.copy2(BASE_SETTINGS_PATH, user_settings_path)
            logger.info(f"[{grag_id}] ✅ 配置文件已复制")
        else:
            logger.warning(f"[{grag_id}] ⚠️ 基础配置文件不存在: {BASE_SETTINGS_PATH}")

        # 步骤3: 构建索引
        log_step(3, TOTAL_STEPS, "构建知识图谱索引 (这可能需要几分钟)", grag_id)
        index_command = f"python -m graphrag index --root {user_path}"

        success, stdout, stderr = run_command_with_progress(
            index_command,
            "索引构建",
            grag_id
        )

        if not success:
            logger.error(f"[{grag_id}] ❌ 索引构建失败")
            await notify_java_backend(
                grag_id=grag_id,
                success=False,
                message="索引构建失败",
                file_saved=filename,
                error=stderr[:500]
            )
            return

        # 步骤4: 提取三元组
        log_step(4, TOTAL_STEPS, "提取三元组数据", grag_id)
        deal_graph_input_dir = os.path.join(user_path, "output")

        extracted_json_path = deal_graph_main(input_dir=deal_graph_input_dir, grag_id=grag_id)

        if not extracted_json_path:
            logger.error(f"[{grag_id}] ❌ 三元组提取失败")
            await notify_java_backend(
                grag_id=grag_id,
                success=False,
                message="图谱创建成功，但三元组提取失败",
                file_saved=filename,
                error="三元组提取返回空路径"
            )
            return

        logger.info(f"[{grag_id}] ✅ 三元组提取完成: {extracted_json_path}")

        # 步骤5: 导入数据到 Neo4j
        log_step(5, TOTAL_STEPS, "导入数据到 Neo4j 数据库", grag_id)

        import_success = insert_neo4j_main(json_file=extracted_json_path)

        if not import_success:
            logger.error(f"[{grag_id}] ❌ 数据库导入失败")
            await notify_java_backend(
                grag_id=grag_id,
                success=False,
                message="图谱创建成功，但数据库导入失败",
                file_saved=filename,
                error="Neo4j导入失败"
            )
            return

        logger.info(f"[{grag_id}] ✅ 数据库导入完成")

        # 【新增】步骤6: 导出节点到CSV
        log_step(6, TOTAL_STEPS, "导出节点到CSV文件（用于实体链接）", grag_id)
        export_success = export_nodes_to_csv(grag_id=grag_id, user_path=user_path)

        if not export_success:
            logger.warning(f"[{grag_id}] ⚠️ 节点导出到CSV失败，但不影响整体流程")
            # 注意：这里不返回，继续通知Java后端（主流程已完成）
        else:
            logger.info(f"[{grag_id}] ✅ 节点导出成功")

        log_step(7, TOTAL_STEPS, "根据csv文件建立密集索引", grag_id)
        retriv_dir = crtDenseRetriever(retriv_dir=os.path.join(user_path, ".retrive"),
                                      file_path=os.path.join(user_path, "nodes_pandas.csv"))
        if retriv_dir:
            logger.info(f"[{grag_id}] ✅ 索引创建成功: {retriv_dir}")
        else:
            logger.warning(f"[{grag_id}] ⚠️ 索引创建失败")

        # 全部成功，通知Java后端
        logger.info(f"[{grag_id}] 🎉 全流程完成！")
        await notify_java_backend(
            grag_id=grag_id,
            success=True,
            message="知识图谱构建、提取、导入及导出全部完成",
            file_saved=filename,
            output_path=os.path.join(user_path, "output"),
            json_extracted=extracted_json_path,
        )

    except Exception as e:
        logger.error(f"[{grag_id}] ❌ 后台任务异常: {e}", exc_info=True)
        await notify_java_backend(
            grag_id=grag_id,
            success=False,
            message="处理过程中发生异常",
            file_saved=filename,
            error=str(e)
        )


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
                    logger.warning(f"{prefix} ⚠️ {clean_line}")

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


def get_neo4j_connector(grag_id: str) -> Neo4jConnector:
    """
    获取或创建 Neo4j 连接实例（带 grag_id 隔离）

    Args:
        grag_id: 图谱ID，用于数据隔离

    Returns:
        Neo4jConnector 实例
    """
    # 使用 grag_id 作为连接池的键
    cache_key = f"connector_{grag_id}"

    if cache_key in db_connections:
        return db_connections[cache_key]

    try:
        connector = Neo4jConnector(
            uri=DEFAULT_NEO4J_CONFIG["uri"],
            username=DEFAULT_NEO4J_CONFIG["username"],
            password=DEFAULT_NEO4J_CONFIG["password"],
            grag_id=grag_id  # 传入 grag_id
        )
        db_connections[cache_key] = connector
        logger.info(f"为图谱 '{grag_id}' 创建新连接")
        return connector
    except Exception as e:
        logger.error(f"创建数据库连接失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"无法连接到数据库 '{grag_id}': {str(e)}"
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



# ============================================================
# 辅助函数：使用大模型生成整合答案
# ============================================================

async def generate_integrated_answer(neo4j_connector: Neo4jConnector, prompt: str) -> str:
    """
    使用大模型生成整合答案

    Args:
        neo4j_connector: Neo4j连接器（用于访问LLM配置）
        prompt: 整合提示词
    Returns:
        整合后的答案
    """
    import httpx

    # 从环境变量或配置中获取LLM API配置
    # 假设使用与ToG相同的LLM配置
    llm_api_url = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
    llm_model = os.getenv("LLM_MODEL", "qwen3:8b")

    try:
        payload = {
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # 较低的温度以获得更准确的答案
                "max_tokens": 2000,
                "top_p": 0.9
            }
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(llm_api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            # 根据API响应格式提取答案
            if isinstance(result, dict):
                answer = result.get("response", "")
            else:
                answer = str(result)

            return answer.strip()

    except httpx.TimeoutException:
        raise Exception("大模型调用超时")
    except Exception as e:
        raise Exception(f"大模型调用失败: {str(e)}")

# ====================================================================================================================================================================================
# /工具函数
# ====================================================================================================================================================================================



# ====================================================================================================================================================================================
# 接口部分
# ====================================================================================================================================================================================


# ============================================================
# CORS跨域测试接口
# ============================================================

@app.get("/CORS_test")
async def index():
    """简单的测试接口，用于验证跨域(CORS)配置是否生效"""
    logger.info("收到 CORS跨域 测试请求")
    return {
        "message": "CORS test successful",
        "status": "ok"
    }


# ============================================================
# ToG查询接口
# ============================================================

@app.post("/query/tog", response_model=ToGQueryResponse)
async def query_with_tog(request: ToGQueryRequest):
    """使用ToG (Think-on-Graph) 方法查询知识图谱"""
    try:
        logger.info("=" * 60)
        logger.info(f"[{request.grag_id}] 🔍 收到ToG查询请求")

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
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return ToGQueryResponse(
                success=False,
                question="",
                answer="",
                execution_time=0,
                grag_id=request.grag_id,
                error_message=error_msg
            )

        logger.info(f"[{request.grag_id}] 💬 问题: {question}")

        # 2. 获取数据库连接（带 grag_id）
        log_step(1, 3, "连接数据库", request.grag_id)
        neo4j_connector = get_neo4j_connector(request.grag_id)
        logger.info(f"[{request.grag_id}] ✅ 数据库连接成功")

        # 3. 创建 ToG 推理引擎
        log_step(2, 3, "初始化ToG推理引擎", request.grag_id)

        dynamic_retriever_path = os.path.join(RETRIEVER_PATH_BASE, request.grag_id, ".retrive")
        tog_reasoning = ToGReasoning(
            neo4j_connector=neo4j_connector,
            llm_model="qwen3:8b",
            api_key="",
            beam_width=request.max_width or 3,
            max_depth=request.max_depth or 10,
            retriever_path=dynamic_retriever_path,
            entity_linking_threshold=ENTITY_LINKING_THRESHOLD
        )
        logger.info(f"[{request.grag_id}] ✅ ToG引擎初始化完成")

        # 4. 执行ToG推理
        log_step(3, 3, "执行ToG推理", request.grag_id)
        result = tog_reasoning.reason(
            question=question,
            max_depth=request.max_depth or 10,
            max_width=request.max_width or 3
        )

        logger.info(f"[{request.grag_id}] ✅ 查询完成，耗时: {result.get('execution_time', 0):.2f}秒")
        logger.info(f"[{request.grag_id}] 📄 答案长度: {len(result.get('answer', ''))} 字符")
        logger.info("=" * 60)

        # 添加 grag_id 到结果
        result["grag_id"] = request.grag_id

        return ToGQueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request.grag_id}] ❌ 查询处理失败: {e}", exc_info=True)
        return ToGQueryResponse(
            success=False,
            question=question if 'question' in locals() else "",
            answer="",
            execution_time=0,
            grag_id=request.grag_id,
            error_message=f"查询处理失败: {str(e)}"
        )


# ============================================================
# GraphRAG查询接口
# ============================================================

@app.post("/query/graphrag", response_model=GraphRAGQueryResponse)
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
# ToG+GraphRAG混合查询接口
# ============================================================

@app.post("/query/tog_grag", response_model=ToGGraphRAGQueryResponse)
async def query_with_tog_graphrag(request: ToGGraphRAGQueryRequest):
    """
    使用ToG和GraphRAG混合方法查询知识图谱
    先分别执行两种查询方法，然后用大模型整合答案
    """
    import time
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info(f"[{request.grag_id}] 🔍 收到ToG+GraphRAG混合查询请求")

        # 1. 解析 Message
        question = None
        if request.messages and len(request.messages) > 0:
            for message in reversed(request.messages):
                if message.role == "user":
                    question = message.content
                    break

        if not question:
            error_msg = "未找到有效的用户问题"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return ToGGraphRAGQueryResponse(
                success=False,
                question="",
                final_answer="",
                tog_answer="",
                graphrag_answer="",
                execution_time=0,
                grag_id=request.grag_id,
                error_message=error_msg
            )

        logger.info(f"[{request.grag_id}] 💬 问题: {question}")

        # 2. 执行 ToG 查询
        log_step(1, 4, "执行ToG查询", request.grag_id)
        try:
            neo4j_connector = get_neo4j_connector(request.grag_id)
            dynamic_retriever_path = os.path.join(RETRIEVER_PATH_BASE, request.grag_id, ".retrive")

            tog_reasoning = ToGReasoning(
                neo4j_connector=neo4j_connector,
                llm_model="qwen3:8b",
                api_key="",
                beam_width=request.max_width or 3,
                max_depth=request.max_depth or 10,
                retriever_path=dynamic_retriever_path,
                entity_linking_threshold=ENTITY_LINKING_THRESHOLD
            )

            tog_result = tog_reasoning.reason(
                question=question,
                max_depth=request.max_depth or 10,
                max_width=request.max_width or 3
            )

            tog_answer = tog_result.get("answer", "")
            tog_success = tog_result.get("success", False)
            logger.info(f"[{request.grag_id}] ✅ ToG查询完成，答案长度: {len(tog_answer)} 字符")

        except Exception as e:
            logger.error(f"[{request.grag_id}] ⚠️ ToG查询失败: {e}")
            tog_answer = ""
            tog_success = False

        # 3. 执行 GraphRAG 查询
        log_step(2, 4, "执行GraphRAG查询", request.grag_id)
        try:
            user_path = os.path.join(GRAPHRAG_ROOT, request.grag_id)
            if not os.path.exists(user_path):
                error_msg = f"目录 {request.grag_id} 不存在，请先创建知识图谱"
                logger.error(f"[{request.grag_id}] ❌ {error_msg}")
                return ToGGraphRAGQueryResponse(
                    success=False,
                    question=question,
                    final_answer="",
                    tog_answer=tog_answer,
                    graphrag_answer="",
                    execution_time=time.time() - start_time,
                    grag_id=request.grag_id,
                    error_message=error_msg
                )

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

            if success:
                graphrag_answer = stdout.strip()
                logger.info(f"[{request.grag_id}] ✅ GraphRAG查询完成，答案长度: {len(graphrag_answer)} 字符")
            else:
                graphrag_answer = ""
                logger.warning(f"[{request.grag_id}] ⚠️ GraphRAG查询失败")

        except Exception as e:
            logger.error(f"[{request.grag_id}] ⚠️ GraphRAG查询异常: {e}")
            graphrag_answer = ""

        # 4. 使用大模型整合答案
        log_step(3, 4, "整合两个答案", request.grag_id)

        if not tog_answer and not graphrag_answer:
            error_msg = "两种查询方法都未返回有效答案"
            logger.error(f"[{request.grag_id}] ❌ {error_msg}")
            return ToGGraphRAGQueryResponse(
                success=False,
                question=question,
                final_answer="",
                tog_answer=tog_answer,
                graphrag_answer=graphrag_answer,
                execution_time=time.time() - start_time,
                grag_id=request.grag_id,
                error_message=error_msg
            )

        # 准备整合提示词
        integration_prompt = f"""你是一个知识图谱查询助手。我使用两种不同的方法查询了同一个问题，现在需要你整合两个答案，给出最准确、最全面的回答。

**问题：** {question}

**方法1 - ToG（思维图谱）的答案：**
{tog_answer if tog_answer else "(未获取到答案)"}

**方法2 - GraphRAG的答案：**
{graphrag_answer if graphrag_answer else "(未获取到答案)"}

请综合以上两个答案，给出一个最终答案。要求：
1. 综合两个答案的优点和补充信息
2. 避免重复
3. 确保回答的准确性和完整性
4. 如果两个答案有冲突，说明你的判断依据
5. 用清晰、结构化的方式(1、2、3...)呈现答案

最终答案："""

        # 5. 调用大模型生成整合答案
        log_step(4, 4, "使用大模型生成最终答案", request.grag_id)
        try:
            # 使用 ToG 推理引擎中的 LLM 生成整合答案
            final_answer = await generate_integrated_answer(
                neo4j_connector=neo4j_connector,
                prompt=integration_prompt
            )
            logger.info(f"[{request.grag_id}] ✅ 整合答案生成完成，长度: {len(final_answer)} 字符")

        except Exception as e:
            logger.error(f"[{request.grag_id}] ❌ 整合答案生成失败: {e}")
            # 如果大模型整合失败，返回较长的那个原始答案
            final_answer = tog_answer if len(tog_answer) > len(graphrag_answer) else graphrag_answer
            logger.warning(f"[{request.grag_id}] ⚠️ 使用原始答案替代整合答案")

        execution_time = time.time() - start_time

        logger.info(f"[{request.grag_id}] ✅ 混合查询完成，总耗时: {execution_time:.2f}秒")
        logger.info("=" * 60)

        return ToGGraphRAGQueryResponse(
            success=True,
            question=question,
            final_answer=final_answer,
            tog_answer=tog_answer,
            graphrag_answer=graphrag_answer,
            grag_id=request.grag_id,
            execution_time=execution_time,
            error_message=None
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[{request.grag_id}] ❌ 混合查询处理失败: {e}", exc_info=True)
        return ToGGraphRAGQueryResponse(
            success=False,
            question=question if 'question' in locals() else "",
            final_answer="",
            tog_answer=tog_answer if 'tog_answer' in locals() else "",
            graphrag_answer=graphrag_answer if 'graphrag_answer' in locals() else "",
            execution_time=execution_time,
            grag_id=request.grag_id,
            error_message=f"查询处理失败: {str(e)}"
        )


# ============================================================
# GraphRAG创建图谱接口 - 立即响应 + 后台处理
# ============================================================

@app.post("/graph/create")
async def create_graph(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        grag_id: str = Form(...)
):
    """
    上传文件并创建GraphRAG知识图谱（异步处理）
    立即返回响应，后台执行创建任务，完成后回调Java后端
    """
    try:
        logger.info("=" * 60)
        logger.info(f"[{grag_id}] 📊 接收到图谱创建请求")

        # 步骤1: 创建用户目录
        user_path = os.path.join(GRAPHRAG_ROOT, grag_id)
        input_dir = os.path.join(user_path, "input")
        os.makedirs(input_dir, exist_ok=True)
        logger.info(f"[{grag_id}] ✅ 目录创建完成: {input_dir}")

        # 步骤2: 保存上传的文件
        file_path = os.path.join(input_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)
        logger.info(f"[{grag_id}] ✅ 文件已保存: {file.filename} ({file_size / 1024:.2f} KB)")

        # 添加后台任务
        background_tasks.add_task(
            create_graph_task,
            file_path=file_path,
            filename=file.filename,
            grag_id=grag_id,
            user_path=user_path,
            input_dir=input_dir
        )

        logger.info(f"[{grag_id}] 📄 后台任务已启动")
        logger.info("=" * 60)

        # 立即返回响应
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,  # 202表示请求已接受，正在处理
            content={
                "success": True,
                "message": "正在创建图谱，请稍候...",
                "status": "processing",
                "grag_id": grag_id,
                "file_saved": file.filename,
                "note": "图谱创建完成后将通过回调接口通知结果"
            }
        )

    except Exception as e:
        logger.error(f"[{grag_id if 'grag_id' in locals() else 'Unknown'}] ❌ 处理失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "请求处理失败",
                "error": str(e),
                "grag_id": grag_id if 'grag_id' in locals() else None
            }
        )


# ====================================================================================================================================================================================
# /接口部分
# ====================================================================================================================================================================================




if __name__ == "__main__":
    import uvicorn

    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "9090"))

    logger.info("=" * 60)
    logger.info("🚀 启动ToG Knowledge Graph API服务器")
    logger.info(f"📍 地址: http://{server_host}:{server_port}")
    logger.info(f"📚 文档: http://{server_host}:{server_port}/docs")
    logger.info(f"🔗 Java回调地址: {JAVA_BACKEND_URL}{JAVA_CALLBACK_PATH}")
    logger.info("=" * 60)

    uvicorn.run(
        "fastapi_server:app",
        host=server_host,
        port=server_port,
        reload=True
    )