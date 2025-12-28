import sys, os

sys.path.append(os.getcwd())
from tqdm import tqdm
from neo4j import GraphDatabase
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
    api_key_name: str,
    base_url: str,
    model: str,
):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""")
        interface_ids = [record["id"] for record in result].data()
        for interface_id in tqdm(interface_ids, desc="interface param descrition"):
            add_interface_param_description(
                driver=driver,
                database=database,
                interface_id=interface_id,
                param_type="input",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )
            add_interface_param_description(
                driver=driver,
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
        + "\n"
        + "你的任务是：将以上接口信息，重写为【面向语义理解的高质量接口描述】。\n"
        + "\n"
        + "重写要求（必须严格遵守）：\n"
        + "1. 只描述接口在业务和能力层面的语义，不做字段级或 JSON 结构说明。\n"
        + "2. 聚焦核心业务数据，忽略与接口核心能力无关的技术性字段(例如响应状态/信息，分页信息等)。\n"
        + "3. 对输入和输出参数进行**业务层面的逻辑分组和抽象**，而不是简单罗列。\n"
        + "4. 对于输入和输出，以业务实体为单位进行描述，并在详细描述业务实体时使用具体参数进行补充\n"
        + "5. **格式模板**：严格遵守以下格式进行输出：\n"
        + "本接口的功能是XXX。\n"
        + "输入包括：\n"
        + "- [业务分组1名称]：使用[参数A]和[参数B]指定，其含义是XXX。\n"
        + "- [业务分组2名称]：使用[参数C]表示，其包含[属性X]、[属性Y]等。\n"
        + "输出包括：\n"
        + "- [业务实体3]：是一个[数据结构](如设备集合)，其关键属性包括[属性P]、[属性Q]等。\n"
        "现在开始重写接口描述：\n"
    )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""")
        interface_ids = [record["id"] for record in result].data()
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
        database="service-cim-2025-12-16",
        api_key_name="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
    rewrite_interface_descriptions(
        uri=url,
        user=user,
        password=password,
        database="service-cim-2025-12-16",
        api_key_name="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
