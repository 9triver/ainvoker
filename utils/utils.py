import pandas as pd
import numpy as np
import json

from typing import Dict, List, Optional, Any
from pandas import DataFrame
from os import getenv
from openai import OpenAI
from neo4j import Driver


CATEGORY_COLS = ["接口一级分类", "接口开发单位", "开发负责人", "联系方式"]
SERVICE_COLS = ["服务名称", "标准化服务名称", "服务描述", "标准化服务描述"]
API_COLS = [
    "接口编码",
    "接口名称",
    "标准化接口名称",
    "接口描述",
    "标准化接口描述",
    "接口生产地址",
    "接口所属系统",
    "是否备选区",
    "调用示例（更新版）",
    "调用问题备注",
    "接口测试结果",
]
PARAM_COLS = [
    "参数类型",
    "参数分类（公共参数/技术参数/业务参数）",
    "参数中文名",
    "参数名",
    "参数字段类型",
    "是否必填（true/false）",
    "格式",
    "是否枚举值",
    "码值对应",
]
EMBED_COLS = [
    "接口一级分类",
    "接口开发单位",
    "开发负责人",
    "接口名称",
    "标准化接口名称",
    "接口描述",
    "标准化接口描述",
    "接口生产地址",
    "接口所属系统",
    "调用问题备注",
]


def read_excel(path: str) -> Dict[str, DataFrame]:
    all_sheets = pd.read_excel(path, sheet_name=None, dtype=str)

    data_frames = {}
    for sheet_name, df in all_sheets.items():
        df = df.ffill().bfill()
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df.columns = df.columns.str.strip().str.replace("\n", "")
        for col in df.columns:
            df[col] = df[col].astype(str).fillna("")
        data_frames[sheet_name] = df

    return data_frames


def to_json(df: DataFrame):
    knowledge_graph = []

    for l1_vals, l1_group in df.groupby(CATEGORY_COLS):
        category_node = dict(zip(CATEGORY_COLS, l1_vals))
        category_node["type"] = "Category"
        category_node["服务"] = []

        for l2_vals, l2_group in l1_group.groupby(SERVICE_COLS):
            service_node = dict(zip(SERVICE_COLS, l2_vals))
            service_node["type"] = "Service"
            service_node["接口"] = []

            for api_vals, api_group in l2_group.groupby(API_COLS):
                api_node = dict(zip(API_COLS, api_vals))
                api_node["type"] = "API"
                api_node["参数"] = [
                    row[PARAM_COLS].to_dict() for _, row in api_group.iterrows()
                ]

                service_node["接口"].append(api_node)

            category_node["服务"].append(service_node)

        knowledge_graph.append(category_node)

    return knowledge_graph


def ask_llm(
    api_key_name: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    top_p: float = 0.7,
    temperature: float = 0.9,
):
    api_key = getenv(api_key_name) if api_key_name else None
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = (
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            top_p=top_p,
            temperature=temperature,
        )
        .choices[0]
        .message.content
    )
    return response


def openai_embedding(embedding_base_url: str, model: str, text: str):
    client = OpenAI(base_url=embedding_base_url, api_key="fake_key")
    embedding = client.embeddings.create(input=text, model=model).data[0].embedding
    return embedding


def get_property(
    driver: Driver, database: str, label: str, id: str, property_name: str
):
    query = f"""
    MATCH (n:{label} {{id: $id}})
    RETURN n.{property_name} AS value
    """
    with driver.session(database=database) as session:
        result = session.run(query, id=id).single()
        return result["value"]


def get_properties(
    driver: Driver, database: str, label: str, id: str, property_names: List[str]
) -> List[Optional[Any]]:
    if not property_names:
        return []

    return_clause = ", ".join([f"n.{prop} AS {prop}" for prop in property_names])
    query = f"""
    MATCH (n:{label} {{id: $id}})
    RETURN {return_clause}
    """

    with driver.session(database=database) as session:
        result = session.run(query, id=id).single()
        if result:
            values = [result.get(prop) for prop in property_names]
            return values
        else:
            return [None] * len(property_names)


def has_property(
    driver: Driver, database: str, label: str, id: str, property_name: str
) -> bool:
    query = f"""
    MATCH (n:{label} {{id: $id}})
    RETURN (n.{property_name} IS NOT NULL) AS has_prop
    """
    with driver.session(database=database) as session:
        result = session.run(query, id=id).single()
        return result["has_prop"] if result else False


def set_property(
    driver: Driver,
    database: str,
    label: str,
    id: str,
    property_name: str,
    property_value,
):
    if property_value is None:
        return None
    if isinstance(property_value, (dict, list)):
        property_value = json.dumps(property_value, ensure_ascii=False)
    query = f"""
    MATCH (n:{label} {{id: $id}})
    SET n.{property_name} = $property_value
    RETURN n.{property_name} AS new_value
    """
    with driver.session(database=database) as session:
        result = session.run(query, id=id, property_value=property_value).single()
        return result["new_value"] if result else None
