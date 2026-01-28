"""
MySQL数据库模型 - 用于存储运维操作记录
"""
import pymysql
from pymysql.err import Error
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger import logger
from core.config import settings


class OperationRecordDB:
    """操作记录数据库管理（MySQL）"""

    def __init__(self):
        """初始化数据库连接"""
        self.connection = None
        self._connect()
        self._init_db()
        logger.info(f"✅ MySQL操作记录数据库初始化完成")

    def _connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = pymysql.connect(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
                charset=settings.mysql_charset,
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"✅ 成功连接到MySQL数据库: {settings.mysql_host}:{settings.mysql_port}")
        except Exception as e:
            logger.error(f"连接MySQL失败: {e}")
            raise

    def _ensure_connection(self):
        """确保连接有效"""
        try:
            self.connection.ping(reconnect=True)
        except Error:
            logger.warning("MySQL连接断开，重新连接...")
            self._connect()

    def _init_db(self):
        """初始化数据库（验证连接和表是否存在）"""
        self._ensure_connection()

        try:
            with self.connection.cursor() as cursor:
                # 验证表是否存在
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'operation_records'
                """, (settings.mysql_database,))

                result = cursor.fetchone()
                if result and result['count'] > 0:
                    logger.info("✅ 数据库表验证完成：operation_records 表存在")
                else:
                    logger.warning("警告：operation_records 表不存在，请先创建该表")
        except Exception as e:
            logger.error(f"数据库初始化验证失败: {e}")
            raise

    def _check_column_exists(self, table_name: str, column_name: str) -> bool:
        """检查表中是否存在某个字段"""
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                sql = """
                    SELECT COUNT(*) as count
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = %s
                """
                cursor.execute(sql, (settings.mysql_database, table_name, column_name))
                result = cursor.fetchone()
                return result and result['count'] > 0
        except Exception as e:
            logger.warning(f"检查字段存在性失败: {e}")
            return False

    def save_record(self, session_id: str, operation: str, image_path: str, summary: str = None) -> int:
        """
        保存操作记录

        Args:
            session_id: 会话ID
            operation: 操作描述
            image_path: 图片路径
            summary: 本次操作的总结描述

        Returns:
            记录ID
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                # 检查 summary 字段是否存在
                has_summary = self._check_column_exists('operation_records', 'summary')

                if has_summary:
                    sql = """
                        INSERT INTO operation_records (session_id, operation, image_path, summary)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (session_id, operation, image_path, summary))
                else:
                    # 如果 summary 字段不存在，则不保存该字段
                    sql = """
                        INSERT INTO operation_records (session_id, operation, image_path)
                        VALUES (%s, %s, %s)
                    """
                    cursor.execute(sql, (session_id, operation, image_path))
                    if summary:
                        logger.warning(f"summary字段不存在，操作总结未保存到数据库: session_id={session_id}")

                self.connection.commit()
                record_id = cursor.lastrowid
                logger.info(f"操作记录保存成功: session_id={session_id}, record_id={record_id}")
                return record_id
        except Exception as e:
            logger.error(f"保存操作记录失败: {e}")
            self.connection.rollback()
            raise

    def get_records_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话的所有操作记录

        Args:
            session_id: 会话ID

        Returns:
            操作记录列表
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                # 检查 summary 字段是否存在
                has_summary = self._check_column_exists('operation_records', 'summary')

                if has_summary:
                    sql = """
                        SELECT id, session_id, operation, image_path, summary, create_time
                        FROM operation_records
                        WHERE session_id = %s
                        ORDER BY create_time ASC
                    """
                else:
                    # 如果 summary 字段不存在，则不查询该字段
                    sql = """
                        SELECT id, session_id, operation, image_path, create_time
                        FROM operation_records
                        WHERE session_id = %s
                        ORDER BY create_time ASC
                    """

                cursor.execute(sql, (session_id,))
                results = cursor.fetchall()

                records = []
                for row in results:
                    record = {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "operation": row["operation"],
                        "image_path": row["image_path"]
                    }
                    # 只有当字段存在时才添加 summary
                    if has_summary:
                        record["summary"] = row.get("summary", "")
                    else:
                        record["summary"] = ""

                    records.append(record)

                if not has_summary:
                    logger.info(f"获取会话记录: session_id={session_id}, count={len(records)} (summary字段不存在)")
                else:
                    logger.info(f"获取会话记录: session_id={session_id}, count={len(records)}")

                return records
        except Exception as e:
            logger.error(f"获取会话记录失败: {e}")
            return []

    def get_session_count(self, session_id: str) -> int:
        """获取会话的记录数量"""
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                sql = """
                    SELECT COUNT(*) as count
                    FROM operation_records
                    WHERE session_id = %s
                """
                cursor.execute(sql, (session_id,))
                result = cursor.fetchone()
                return result["count"] if result else 0
        except Exception as e:
            logger.error(f"获取记录数量失败: {e}")
            return 0

    def delete_records_by_session(self, session_id: str) -> int:
        """
        删除会话的所有操作记录

        Args:
            session_id: 会话ID

        Returns:
            删除的记录数量
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                sql = """
                    DELETE FROM operation_records
                    WHERE session_id = %s
                """
                cursor.execute(sql, (session_id,))
                deleted_count = cursor.rowcount
                self.connection.commit()
                logger.info(f"删除会话记录成功: session_id={session_id}, deleted_count={deleted_count}")
                return deleted_count
        except Exception as e:
            logger.error(f"删除会话记录失败: {e}")
            self.connection.rollback()
            raise

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("MySQL连接已关闭")


# 全局数据库实例（延迟初始化）
_operation_db_instance: Optional[OperationRecordDB] = None


def get_operation_db() -> OperationRecordDB:
    """获取操作记录数据库单例"""
    global _operation_db_instance
    if _operation_db_instance is None:
        _operation_db_instance = OperationRecordDB()
    return _operation_db_instance
