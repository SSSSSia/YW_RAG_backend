"""
GraphRAG 三元组提取工具
每个 output 文件夹作为独立知识库，通过 grag_id 标识
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any
from datetime import datetime


class GraphRAGExtractor:
    """从 GraphRAG 输出中提取三元组并标记 grag_id"""

    def __init__(self, input_dir: str, output_dir: str = None, grag_id: str = ""):
        """
        初始化提取器

        Args:
            input_dir: GraphRAG 的 output 文件夹路径
            output_dir: 输出目录，默认为 input_dir/extracted_data
            grag_id: 知识库唯一标识符，必须提供
        """
        if not grag_id or not grag_id.strip():
            raise ValueError("grag_id 不能为空！每个知识库必须有唯一标识")

        self.input_dir = input_dir
        self.grag_id = grag_id.strip()

        # 默认输出到 input_dir 下的 extracted_data 目录
        if output_dir is None:
            self.output_dir = os.path.join(input_dir, "extracted_data")
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def load_entities(self) -> pd.DataFrame:
        """加载实体数据"""
        file_path = os.path.join(self.input_dir, "entities.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到实体文件: {file_path}")
        return pd.read_parquet(file_path)

    def load_relationships(self) -> pd.DataFrame:
        """加载关系数据"""
        file_path = os.path.join(self.input_dir, "relationships.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到关系文件: {file_path}")
        return pd.read_parquet(file_path)

    def extract_entities(self, entities_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        提取实体信息，每个实体都标记 grag_id

        Returns:
            实体字典，key 为 entity_id
        """
        entity_dict = {}
        for _, row in entities_df.iterrows():
            entity_id = str(row.get('id', row.get('human_readable_id', '')))
            entity_dict[entity_id] = {
                'id': entity_id,
                'name': str(row.get('name', row.get('title', ''))),
                'type': str(row.get('type', 'ENTITY')),
                'description': str(row.get('description', '')),
                'degree': int(row.get('degree', 0)) if pd.notna(row.get('degree')) else 0,
                'community_ids': self._safe_list(row.get('community_ids', [])),
                'text_unit_ids': self._safe_list(row.get('text_unit_ids', [])),
                'grag_id': self.grag_id  # 🔴 关键：标记知识库ID
            }
        return entity_dict

    def extract_triples(self,
                        relationships_df: pd.DataFrame,
                        entity_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取三元组信息，每条关系都标记 grag_id

        Returns:
            三元组列表
        """
        triples = []
        for idx, row in relationships_df.iterrows():
            source_id = str(row.get('source', ''))
            target_id = str(row.get('target', ''))

            # 获取实体名称
            source_name = entity_dict.get(source_id, {}).get('name', source_id)
            target_name = entity_dict.get(target_id, {}).get('name', target_id)

            triple = {
                'id': f"rel_{idx}",
                'subject': source_name,
                'subject_id': source_id,
                'predicate': str(row.get('type', row.get('description', 'RELATED_TO'))),
                'object': target_name,
                'object_id': target_id,
                'weight': float(row.get('weight', 1.0)) if pd.notna(row.get('weight')) else 1.0,
                'description': str(row.get('description', '')),
                'source_degree': int(row.get('source_degree', 0)) if pd.notna(row.get('source_degree')) else 0,
                'target_degree': int(row.get('target_degree', 0)) if pd.notna(row.get('target_degree')) else 0,
                'rank': int(row.get('rank', 0)) if pd.notna(row.get('rank')) else 0,
                'grag_id': self.grag_id  # 🔴 关键：标记知识库ID
            }
            triples.append(triple)
        return triples

    def _safe_list(self, value):
        """安全转换为列表"""
        import numpy as np
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist() if value.size > 0 else []
        if pd.isna(value) and not hasattr(value, '__len__'):
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, str):
            return [value]
        return []

    def run(self) -> str:
        """
        执行提取流程

        Returns:
            生成的 JSON 文件路径
        """
        print(f"[INFO] 开始提取知识库: {self.grag_id}")
        print(f"[INFO] 输入目录: {self.input_dir}")

        # 加载数据
        entities_df = self.load_entities()
        relationships_df = self.load_relationships()
        print(f"[INFO] 加载 {len(entities_df)} 个实体, {len(relationships_df)} 条关系")

        # 提取数据
        entities = self.extract_entities(entities_df)
        triples = self.extract_triples(relationships_df, entities)

        # 构建输出数据
        output_path = os.path.join(self.output_dir, "graph_data.json")
        data = {
            'metadata': {
                'extraction_time': datetime.now().isoformat(),
                'source_directory': self.input_dir,
                'grag_id': self.grag_id,
                'entity_count': len(entities),
                'triple_count': len(triples)
            },
            'entities': entities,
            'triples': triples
        }

        # 保存 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] 数据已提取至: {output_path}")
        return output_path


def main(input_dir: str, grag_id: str) -> str | None:
    """
    主函数：提取数据并返回文件路径

    Args:
        input_dir: GraphRAG 的 output 文件夹路径
        grag_id: 知识库唯一标识符

    Returns:
        str: 成功时返回生成的 JSON 文件的绝对路径
        None: 失败时返回 None
    """
    try:
        if not grag_id or not grag_id.strip():
            print("[ERROR] grag_id 不能为空")
            return None

        extractor = GraphRAGExtractor(input_dir=input_dir, grag_id=grag_id)
        output_path = extractor.run()
        return os.path.abspath(output_path)

    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] 提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 测试调用
    result_path = main(
        input_dir="./output",
        grag_id="knowledge_base_20250112_001"
    )

    if result_path:
        print(f"\n✅ 生成文件: {result_path}")
    else:
        print("\n❌ 提取失败")