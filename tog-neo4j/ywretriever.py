from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.documents import Document
import pandas as pd
from typing import List


def crtDenseRetriever(retriv_dir: str = "D:\\CODE_FILE\\CODE_PYTHON\\YW_RAG_backend\\tog-neo4j\\.retrive\\ywcorom",
                      file_path: str = "D:\\CODE_FILE\\CODE_PYTHON\\YW_RAG_backend\\tog-neo4j\\nodes_pandas.csv"):
    """
    使用LangChain建立密集索引
    """
    model_name = "iic/nlp_corom_sentence-embedding_chinese-base"
    embeddings = ModelScopeEmbeddings(model_id=model_name)

    df = pd.read_csv(file_path)

    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["name"],
            metadata={"id": row["id"]}
        )
        documents.append(doc)

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vectorstore.save_local(retriv_dir)
    print(f"✓ 索引创建成功: {retriv_dir}")
    return retriv_dir, model_name


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


def entity_linking(retriever_obj: Retriever, entities: list[str], threshold: float = 15) -> list[str]:
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

            # 根据阈值判断是否匹配成功
            if score <= threshold:
                status = "✓"
                match_type = "匹配"
            else:
                status = "✗"
                match_type = "超阈值"

            # 格式化输出
            print(f"{status} {ent:20s} -> {doc.page_content:20s} [{match_type}] (距离: {score:.2f})")

    print("=" * 60 + "\n")
    return results


if __name__ == "__main__":
    # 创建索引
    crtDenseRetriever()

    # 加载检索器
    my_retriever = Retriever(
        retrievel_type="dense",
        retriever_version="D:/CODE_FILE/CODE_PYTHON/YW_RAG_backend/tog-neo4j/.retrive/ywcorom"
    )

    # 测试数据
    test_entities = ["GHOST镜像", "复制文件", "激活期限"]

    print("\n" + "=" * 60)
    print("【开始实体链接测试】")
    print("=" * 60)

    linked_results = entity_linking(my_retriever, test_entities, threshold=15.0)

    print("\n" + "=" * 60)
    print("【最终结果】")
    print("-" * 60)
    for i, res in enumerate(linked_results, 1):
        print(f"{i}. {res}")
    print("=" * 60)