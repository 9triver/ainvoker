NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "service-list"

import json, os
from neo4j import GraphDatabase, Driver
from tqdm import tqdm
from service_list.utils.utils import ask_llm
from loguru import logger


def get_parameter_ids(driver: Driver, database: str):
    param_ids = []
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (n:Parameter)
            WITH n
            RETURN n.id AS id
            """
        )
        param_ids = [record["id"] for record in result]
        return param_ids


def get_parameter_full_name(driver: Driver, database: str, param_id: str):
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (p:Parameter {id: $param_id})
            RETURN p.chinese_name as chinese_name, p.name as name
            """,
            param_id=param_id,
        ).data()
        return "{chinese_name}/{name}".format(
            chinese_name=result[0]["chinese_name"], name=result[0]["name"]
        )


def get_interfaces_description_to_param(driver: Driver, database: str, param_id: str):
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (p:Parameter {id: $param_id})<-[:OUTPUT_FROM_INTERFACE]-(i:Interface)
            RETURN i.description as description
            """,
            param_id=param_id,
        ).data()
        return [record["description"] for record in result]


def get_interfaces_description_from_param(driver: Driver, database: str, param_id: str):
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (p:Parameter {id: $param_id})-[:INPUT_TO_INTERFACE]->(i:Interface)
            RETURN i.description as description
            """,
            param_id=param_id,
        ).data()
        return [record["description"] for record in result]


def get_param_description(prompt: str):
    response = ask_llm(
        api_key_name="CHATGLM_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4.5-flash",
        system_prompt="遵循用户指令，完成用户任务",
        user_prompt=prompt,
    )
    logger.info(response)
    return response


def write_param_description(
    driver: Driver, database: str, param_id: str, param_description: str
):
    with driver.session(database=database) as session:
        session.run(
            """
            MATCH (p:Parameter {id: $param_id})
            SET p.description = $param_description
            """,
            param_id=param_id,
            param_description=param_description,
        )
    return


output_path = "./data/service/param_descriptions.json"


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    param_ids = get_parameter_ids(driver=driver, database=NEO4J_DATABASE)

    param_descriptions = {}
    if os.path.exists(path=output_path):
        with open(file=output_path, mode="r", encoding="utf-8") as f:
            param_descriptions = json.load(fp=f)
    else:
        for param_id in tqdm(param_ids, desc="process params"):
            to_interfaces = get_interfaces_description_from_param(
                driver=driver, database=NEO4J_DATABASE, param_id=param_id
            )
            from_interfaces = get_interfaces_description_to_param(
                driver=driver, database=NEO4J_DATABASE, param_id=param_id
            )
            full_name = get_parameter_full_name(
                driver=driver, database=NEO4J_DATABASE, param_id=param_id
            )
            prompt = "一个名为'{full_name}'参数，它在下面这些接口中被用于输入/输出参数，输入参数接口描述：{to_interfaces};输出参数接口描述：{from_interfaces}。根据你的理解，在不提及接口信息的前提下，返回给我一段关于参数的**纯文本描述**，辅助用户理解该参数含义：".format(
                full_name=full_name,
                to_interfaces=json.dumps(obj=to_interfaces, ensure_ascii=False),
                from_interfaces=json.dumps(obj=from_interfaces, ensure_ascii=False),
            )
            param_descriptions[param_id] = {
                "full_name": full_name,
                "prompt": prompt,
                "description": "",
                "from_interfaces": from_interfaces,
                "to_interfaces": to_interfaces,
            }

        for param_id in tqdm(param_ids, desc="get param descriptions"):
            param_descriptions[param_id]["description"] = get_param_description(
                prompt=param_descriptions[param_id]["prompt"]
            )
            with open(file=output_path, mode="w", encoding="utf-8") as f:
                json.dump(obj=param_descriptions, fp=f, ensure_ascii=False, indent=4)

    # write into database
    for param_id in tqdm(param_ids, desc="write params"):
        param_description = param_descriptions[param_id]["description"]
        write_param_description(
            driver=driver,
            database=NEO4J_DATABASE,
            param_id=param_id,
            param_description=param_description,
        )


if __name__ == "__main__":
    main()
