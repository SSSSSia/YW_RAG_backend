from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.documents import Document
import pandas as pd
import os
from typing import List


def crtDenseRetriever(retriv_dir: str, file_path: str):
    """
    使用LangChain建立密集索引

    Args:
        retriv_dir: 向量索引保存目录 (例如: ../graphrag/{grag_id}/..retrive)
        file_path: 节点CSV文件路径 (例如: ../graphrag/{grag_id}/nodes_pandas.csv)

    Returns:
        retriv_dir：向量索引保存目录 (例如: ../graphrag/{grag_id}/..retrive)
    """
    model_name = "iic/nlp_corom_sentence-embedding_chinese-base"
    embeddings = ModelScopeEmbeddings(model_id=model_name)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV文件不存在: {file_path}")

    df = pd.read_csv(file_path)

    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["name"],
            metadata={"id": row["id"]}
        )
        documents.append(doc)

    # 如果目录不存在，创建目录
    os.makedirs(retriv_dir, exist_ok=True)

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vectorstore.save_local(retriv_dir)
    print(f"✓ 索引创建成功: {retriv_dir}")
    return retriv_dir


class LangChainDenseRetriever:
    def __init__(self, vectorstore, top_k: int = 5):
        self.vectorstore = vectorstore
        self.top_k = top_k

    @classmethod
    def load(cls, retriever_version: str, top_k: int = 5):
        embeddings = ModelScopeEmbeddings(model_id="iic/nlp_corom_sentence-embedding_chinese-base")
        vectorstore = FAISS.load_local(
            retriever_version,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return cls(vectorstore, top_k)

    def search_with_score(self, query: str):
        return self.vectorstore.similarity_search_with_score(query, k=self.top_k)


class Retriever:
    def __init__(self, retrievel_type: str, retriever_version: str):
        if retrievel_type == "dense":
            self.retriever = LangChainDenseRetriever.load(retriever_version)
        else:
            raise ValueError("Invalid retriever type")

    def retrieve(self, query: str):
        return self.retriever.search_with_score(query)


def entity_linking(retriever_obj: Retriever, entities: list[str], threshold: float = 10) -> list[str]:
    """
    实体链接 - 简洁版输出
    """
    results = []

    print("\n" + "=" * 60)
    print("【实体链接结果】")
    print("-" * 60)

    for ent in entities:
        search_res = retriever_obj.retrieve(ent)
        if search_res:
            doc, score = search_res[0]
            results.append(doc.page_content)

            if score <= threshold:
                status = "✓"
                match_type = "匹配"
            else:
                status = "✗"
                match_type = "超阈值"

            print(f"{status} {ent:20s} -> {doc.page_content:20s} [{match_type}] (距离: {score:.2f})")

    print("=" * 60 + "\n")
    return results
