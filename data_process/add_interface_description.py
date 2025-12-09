import sys, os

sys.path.append(os.getcwd())
from tqdm import tqdm
from neo4j import GraphDatabase
from tools.service import ServiceTools
from utils.utils import (
    ask_llm,
    has_property,
    set_property,
    get_property,
    get_properties,
)
from loguru import logger


def add_interface_param_description(
    driver,
    database: str,
    interface_id: str,
    param_type: str,
    api_key_name: str,
    base_url: str,
    model: str,
):
    property_name = f"{param_type}_description"
    detail_key = f"{param_type}_parameters"
    require_prompt = """要求：
    1. 明确输入实体并进行描述。
    2. 复杂的实体需要展开后描述其内部实体。
    3. 忽略重要性不强的实体、繁琐的格式说明等等。"""
    if param_type == "input":
        prompt = f"根据上面的接口的json格式，写一段描述接口所需输入的描述。\n{require_prompt}\n直接返回接口输入描述："
    else:
        prompt = f"根据上面的接口的json格式，写一段描述接口所需输出的描述。\n{require_prompt}\n直接返回接口输出描述："

    if has_property(
        driver=driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name=property_name,
    ):
        return

    interface_details = {}
    for key in [
        "name",
        "description",
        "standard_name",
        "example",
        detail_key,
    ]:
        value = get_property(
            driver=driver,
            database=database,
            label="Interface",
            id=interface_id,
            property_name=key,
        )
        if value:
            interface_details[key] = value
    user_prompt = "{interface_details}\n{prompt}".format(
        interface_details=interface_details,
        prompt=prompt,
    )
    input_description = ask_llm(
        api_key_name=api_key_name,
        base_url=base_url,
        model=model,
        system_prompt="遵循用户指令，完成用户任务",
        user_prompt=user_prompt,
    )
    logger.info(input_description)
    set_property(
        driver=driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name=property_name,
        property_value=input_description,
    )
    return


def add_interface_param_descriptions(
    uri: str,
    user: str,
    password: str,
    database: str,
    embedding_base_url: str,
    embedding_model: str,
    api_key_name: str,
    base_url: str,
    model: str,
):
    service_tool = ServiceTools(
        uri=uri,
        user=user,
        password=password,
        database=database,
        base_url=embedding_base_url,
        embedding_model=embedding_model,
    )
    with service_tool.driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""")
        interface_ids = [record["id"] for record in result]
        for interface_id in tqdm(interface_ids, desc="interface param descrition"):
            add_interface_param_description(
                driver=service_tool.driver,
                database=database,
                interface_id=interface_id,
                param_type="input",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )
            add_interface_param_description(
                driver=service_tool.driver,
                database=database,
                interface_id=interface_id,
                param_type="output",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )


def rewrite_interface_descriptions(
    uri: str,
    user: str,
    password: str,
    database: str,
    api_key_name: str,
    base_url: str,
    model: str,
):
    prompt_template = (
        "接口名称: {interface_name}\n"
        + "接口原始描述: {interface_description}\n"
        + "接口输入描述: {interface_input_description}\n"
        + "接口输出描述: {interface_output_description}\n"
        + "接口调用示例: {interface_example}\n"
        + "结合上面信息，重写一个高质量的接口描述，要求如下:\n"
        + "1. 描述清楚接口的功能、输入和输出\n"
        + "2. 忽略无用的参数、繁琐的格式说明等等\n"
        + "3. 参考模版: 本接口的功能是XXX。输入包括：有用参数及其描述。输出包括：有用参数及其描述。参数描述可以这样写：XXX: 其含义是XXX，其包括XXX属性，XXX其他参数\n"
        + "重写的高质量接口描述: "
    )
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""")
        interface_ids = [record["id"] for record in result]
        for interface_id in tqdm(interface_ids, desc="interface descrition"):
            if has_property(
                driver=driver,
                database=database,
                label="Interface",
                id=interface_id,
                property_name="llm_description",
            ):
                continue

            interface_infomations = get_properties(
                driver=driver,
                database=database,
                label="Interface",
                id=interface_id,
                property_names=[
                    "name",
                    "description",
                    "input_description",
                    "output_description",
                    "example",
                ],
            )
            user_prompt = prompt_template.format(
                interface_name=interface_infomations[0].strip(),
                interface_description=interface_infomations[1].strip(),
                interface_input_description=interface_infomations[2].strip(),
                interface_output_description=interface_infomations[3].strip(),
                interface_example=interface_infomations[4].strip(),
            )
            interface_rewrite_description = ask_llm(
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
                system_prompt="遵循用户指令，完成用户任务",
                user_prompt=user_prompt,
            )
            logger.info(interface_rewrite_description)
            set_property(
                driver=driver,
                database=database,
                label="Interface",
                id=interface_id,
                property_name="llm_description",
                property_value=interface_rewrite_description,
            )


if __name__ == "__main__":
    url = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    add_interface_param_descriptions(
        uri=url,
        user=user,
        password=password,
        database="service-cim",
        embedding_base_url=os.getenv("EMBED_BASE_URL"),
        embedding_model="text-embedding-qwen3-embedding-8b",
        api_key_name="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
    rewrite_interface_descriptions(
        uri=url,
        user=user,
        password=password,
        database="service-cim",
        api_key_name="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
