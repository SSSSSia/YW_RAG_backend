from neo4j import GraphDatabase
import pandas as pd

uri = "bolt://localhost:7687"
user = "neo4j"
password = "jbh966225"

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run("MATCH (n) RETURN elementId(n) AS id, COALESCE(n.name, '') AS name")
    # 将结果直接转换为 DataFrame
    df = pd.DataFrame([record.data() for record in result])

    # 保存为 CSV，index=False 表示不保存行号
    df.to_csv("nodes_pandas.csv", index=False, encoding='utf-8')

print("导出完成")

driver.close()
