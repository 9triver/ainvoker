NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "service-cim"

LABEL_ATTR_REMAP = {
    "Interface": {
        "attr1": "description",
        "attr2": "code",
        "attr3": "standard_name",
        "attr4": "standard_description",
        "attr5": "production_url",
    },
    "Parameter": {
        "attr1": "chinese_name",
        "attr2": "field_type",
        "attr3": "required",
        "attr4": "format",
        "attr5": "is_enum",
        "attr6": "code_mapping",
    },
    "Person": {
        "attr1": "contact_info",
    },
    "Service": {
        "attr1": "description",
        "attr2": "standard_name",
        "attr3": "standard_description",
    },
}

from neo4j import GraphDatabase
from collections import defaultdict


def get_all_nodes_label(driver, database):
    """
    获取数据库中所有节点的 elementId 和其第一个 Label。
    假设每个节点只有一个 Label。
    """
    node_label_map = {}
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (n)
            WITH n, labels(n) AS lbs
            WHERE size(lbs) > 0
            RETURN elementId(n) AS id, lbs[0] AS label
        """
        )
        for record in result:
            node_id = record["id"]
            label = record["label"]
            if isinstance(label, str):
                node_label_map[node_id] = label
    return node_label_map


def rename_attributes_for_nodes(driver, node_ids, attr_map, database):
    """
    对指定的一批节点，根据 attr_map 将旧属性名改为新属性名。
    例如：{"attr1": "name"} 表示将 attr1 的值赋给 name，并删除 attr1。
    """
    set_clauses = []
    remove_clauses = []
    for old_key, new_key in attr_map.items():
        set_clauses.append(f"n.`{new_key}` = n.`{old_key}`")
        remove_clauses.append(f"n.`{old_key}`")

    if not set_clauses:
        return

    set_clause_str = ", ".join(set_clauses)
    remove_clause_str = ", ".join(remove_clauses)

    query = f"""
    UNWIND $node_ids AS node_id
    MATCH (n) WHERE elementId(n) = node_id
    SET {set_clause_str}
    REMOVE {remove_clause_str}
    """

    with driver.session(database=database) as session:
        session.run(query, node_ids=node_ids)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        print("正在读取所有节点的 elementId 和 Label...")
        node_label_map = get_all_nodes_label(driver, NEO4J_DATABASE)
        print(f"共读取 {len(node_label_map)} 个节点")

        # 按 Label 分组节点 ID
        label_to_node_ids = defaultdict(list)
        for node_id, label in node_label_map.items():
            if label in LABEL_ATTR_REMAP:
                label_to_node_ids[label].append(node_id)

        # 对每个 Label 执行属性重命名
        for label, node_ids in label_to_node_ids.items():
            if not node_ids:
                continue
            attr_map = LABEL_ATTR_REMAP[label]
            print(f"正在处理 Label='{label}' 的 {len(node_ids)} 个节点...")
            rename_attributes_for_nodes(driver, node_ids, attr_map, NEO4J_DATABASE)

        print("✅ 所有属性重命名完成！")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
