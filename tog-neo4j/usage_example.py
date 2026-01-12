"""
使用示例和测试代码
演示如何调用 FastAPI 接口进行查询
"""
import requests
import json

# API 基础地址
BASE_URL = "http://localhost:8000"


def test_health_check():
    """测试健康检查接口"""
    print("=== 测试健康检查 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_query(question: str):
    """测试主查询接口"""
    print(f"=== 测试查询: {question} ===")

    payload = {
        "question": question,
        "max_depth": 3,
        "max_width": 3,
        "enable_pruning": True,
        "temperature": 0.0
    }

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {response.status_code}")
    result = response.json()

    print(f"\n问题: {result['question']}")
    print(f"答案: {result['answer']}")
    print(f"执行时间: {result['execution_time']:.2f}秒")
    print(f"检索到的实体: {result['retrieved_entities']}")
    print(f"推理路径: {json.dumps(result['reasoning_path'], indent=2, ensure_ascii=False)}")
    print()


def test_search_entities(keyword: str):
    """测试实体搜索"""
    print(f"=== 搜索实体: {keyword} ===")

    response = requests.post(
        f"{BASE_URL}/search_entities",
        params={"keyword": keyword, "limit": 5}
    )

    print(f"状态码: {response.status_code}")
    result = response.json()

    print(f"找到 {result['count']} 个实体:")
    for entity in result['entities']:
        print(f"  - {entity['entity_name']} ({entity['entity_labels']})")
    print()


def test_get_neighbors(entity_name: str):
    """测试获取邻居"""
    print(f"=== 获取实体邻居: {entity_name} ===")

    response = requests.post(
        f"{BASE_URL}/get_neighbors",
        params={"entity_name": entity_name, "depth": 1}
    )

    print(f"状态码: {response.status_code}")
    result = response.json()

    print(f"找到 {len(result['neighbors'])} 个邻居:")
    for neighbor in result['neighbors'][:5]:  # 只显示前5个
        print(
            f"  - {neighbor.get('source_entity', '')} --[{neighbor.get('relation', '')}]--> {neighbor.get('target_entity', '')}")
    print()


def test_execute_cypher(query: str):
    """测试执行 Cypher 查询"""
    print(f"=== 执行 Cypher 查询 ===")
    print(f"查询: {query}")

    response = requests.post(
        f"{BASE_URL}/execute_cypher",
        params={"cypher_query": query}
    )

    print(f"状态码: {response.status_code}")
    result = response.json()

    print(f"返回 {result['count']} 条记录:")
    for record in result['results'][:3]:  # 只显示前3条
        print(f"  {record}")
    print()


if __name__ == "__main__":
    # 测试示例

    # 1. 健康检查
    test_health_check()

    # 2. 搜索实体
    test_search_entities("北京")

    # 3. 获取实体邻居
    test_get_neighbors("北京")

    # 4. 执行简单查询
    test_execute_cypher("MATCH (n) RETURN n.name LIMIT 5")

    # 5. 主查询功能(自然语言)
    questions = [
        "北京有哪些著名景点?",
        "谁是苹果公司的CEO?",
        "中国的首都在哪里?"
    ]

    for question in questions:
        test_query(question)

# ============================================
# 使用 curl 命令测试的示例
# ============================================

"""
# 1. 健康检查
curl http://localhost:8000/health

# 2. 搜索实体
curl -X POST "http://localhost:8000/search_entities?keyword=北京&limit=5"

# 3. 获取邻居
curl -X POST "http://localhost:8000/get_neighbors?entity_name=北京&depth=1"

# 4. 主查询(自然语言)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "北京有哪些著名景点?",
    "max_depth": 3,
    "max_width": 3,
    "enable_pruning": true,
    "temperature": 0.0
  }'

# 5. 执行 Cypher 查询
curl -X POST "http://localhost:8000/execute_cypher?cypher_query=MATCH%20(n)%20RETURN%20n.name%20LIMIT%205"
"""