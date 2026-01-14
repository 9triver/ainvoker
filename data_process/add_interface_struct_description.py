import sys, os, json, random

sys.path.append(os.getcwd())
from tqdm import tqdm
from neo4j import GraphDatabase
from utils.utils import (
    ask_llm,
    has_property,
    get_property,
    get_properties,
    set_property,
)

from json_repair import loads
from loguru import logger


def convert_interface_llm_description_to_struct(
    driver,
    database: str,
    interface_id: str,
    param_type: str,
    api_key_name: str,
    base_url: str,
    model: str,
):
    property_name = f"{param_type}_entities"
    if has_property(
        driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name=property_name,
    ):
        return

    interface_name = get_property(
        driver=driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name="name",
    )
    llm_description = get_property(
        driver=driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name=f"llm_description",
    )
    require_prompt = """要求：
    1. 将接口描述中的实体提取为下面的json结构：
    {
        \"业务实体名\": \"业务实体描述\",
        ...
    }
    
    2. 以业务实体为单位进行描述，并在详细描述业务实体时使用具体参数进行补充。
    3. 输入/输出描述中的业务实体已经经过了**业务层面的逻辑分组和抽象**，不要再次拆分开来，将输入/输出描述切分后照抄或转述即可。
    4. 业务实体名和业务实体描述 **必须是文本字符串**，禁止嵌套字符串，只能有一层键值对。
    """
    if param_type == "input":
        system_prompt = f"根据上面提供的接口的相关信息，根据要求将**输入描述**结构化，**忽略输出描述**。\n{require_prompt}\n直接返回接口输入描述的json："
    else:
        system_prompt = f"根据上面提供的接口的相关信息，根据要求将**输出描述**结构化，**忽略输出描述**。\n{require_prompt}\n直接返回接口输出描述的json："
    interface_infomation = f"一、接口名称: {interface_name.strip()}\n\n二、接口描述{llm_description.strip()}"

    user_prompt = "{interface_infomation}\n{system_prompt}".format(
        interface_infomation=interface_infomation,
        system_prompt=system_prompt,
    )
    response = ask_llm(
        api_key_name=api_key_name,
        base_url=base_url,
        model=model,
        system_prompt="遵循用户指令，完成用户任务",
        user_prompt=user_prompt,
    )
    parameters = loads(json_str=response)
    if isinstance(parameters, list):
        parameters = parameters[-1]
    if not isinstance(parameters, dict):
        return None
    logger.info(
        f"\nentities:\n{json.dumps(obj=parameters, ensure_ascii=False, indent=2)}"
    )
    return set_property(
        driver=driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name=property_name,
        property_value=parameters,
    )


def convert_interface_llm_descriptions_to_struct(
    uri: str,
    user: str,
    password: str,
    database: str,
    api_key_name: str,
    base_url: str,
    model: str,
):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""").data()
        interface_ids = [record["id"] for record in result]
        for interface_id in tqdm(interface_ids, desc="interface param descrition"):

            llm_description: str = get_property(
                driver=driver,
                database=database,
                label="Interface",
                id=interface_id,
                property_name=f"llm_description",
            )
            llm_function_description = llm_description.split("输入包括", 1)[0].strip()
            if not llm_function_description:
                logger.info(llm_description)
            set_property(
                driver=driver,
                database=database,
                label="Interface",
                id=interface_id,
                property_name="llm_function_description",
                property_value=llm_function_description,
            )

            convert_interface_llm_description_to_struct(
                driver=driver,
                database=database,
                interface_id=interface_id,
                param_type="input",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )
            convert_interface_llm_description_to_struct(
                driver=driver,
                database=database,
                interface_id=interface_id,
                param_type="output",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )


def write_entities_into_database(
    uri: str,
    user: str,
    password: str,
    database: str,
):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""").data()
        interface_ids = [record["id"] for record in result]
        for interface_id in tqdm(interface_ids):
            input_entities, output_entities, llm_function_description = get_properties(
                driver=driver,
                database=database,
                label="Interface",
                id=interface_id,
                property_names=[
                    "input_entities",
                    "output_entities",
                    "llm_function_description",
                ],
            )
            input_entities = loads(json_str=input_entities)
            for entity_name, entity_description in input_entities.items():
                session.run(
                    """
                    MATCH (i:Interface {id: $interface_id})
                    MERGE (e:InputEntity {name: $name})
                    ON CREATE SET
                        e.id = $entity_id,
                        e.description = $description,
                        e.interface_llm_function_description = $interface_llm_function_description
                    MERGE (e)-[:INPUT_ENTITY]->(i)
                    """,
                    interface_id=interface_id,
                    name=entity_name,
                    description=entity_description,
                    interface_llm_function_description=llm_function_description,
                    entity_id=f"Entity_{random.randint(10_000_000, 99_999_999)}",
                )
            output_entities = loads(json_str=output_entities)
            for entity_name, entity_description in output_entities.items():
                session.run(
                    """
                    MATCH (i:Interface {id: $interface_id})
                    MERGE (e:OutputEntity {name: $name})
                    ON CREATE SET
                        e.id = $entity_id,
                        e.description = $description,
                        e.interface_llm_function_description = $interface_llm_function_description
                    MERGE (i)-[:OUTPUT_ENTITY]->(e)
                    """,
                    interface_id=interface_id,
                    name=entity_name,
                    description=entity_description,
                    interface_llm_function_description=llm_function_description,
                    entity_id=f"Entity_{random.randint(10_000_000, 99_999_999)}",
                )


if __name__ == "__main__":
    url = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    # convert_interface_llm_descriptions_to_struct(
    #     uri=url,
    #     user=user,
    #     password=password,
    #     database="service-cim-2026-01-10",
    #     api_key_name="MIMO_API_KEY",
    #     base_url="https://api.xiaomimimo.com/v1",
    #     model="mimo-v2-flash",
    # )
    write_entities_into_database(
        uri=url,
        user=user,
        password=password,
        database="service-cim-2026-01-10",
    )
