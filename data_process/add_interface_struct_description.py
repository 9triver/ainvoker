import sys, os, json

sys.path.append(os.getcwd())
from tqdm import tqdm
from tools.service import ServiceTools
from utils.utils import ask_llm, has_property, get_property, set_property

from json_repair import loads
from loguru import logger


def convert_interface_param_description_to_struct(
    driver,
    database: str,
    interface_id: str,
    param_type: str,
    api_key_name: str,
    base_url: str,
    model: str,
):
    property_name = f"{param_type}_params"
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
    example = get_property(
        driver=driver,
        database=database,
        label="Interface",
        id=interface_id,
        property_name="example",
    )
    param_description = get_property(
        driver=driver,
        database=database,
        interface_id=interface_id,
        property_name=f"{param_type}_description",
    )
    require_prompt = """要求：
    1. 忽略没有具体含义的参数，关注可以后续利用的参数实体。
    2. 将参数描述中的实体提取为下面的json结构：
    {
        "中文参数名(english_param_name): \"param_description\",
        ...
    }
    """
    if param_type == "input":
        param_name = "输入参数描述"
        system_prompt = f"根据上面提供的接口的相关信息，根据要求将**输入参数**的描述结构化。\n{require_prompt}\n直接返回接口输入描述的json："
    else:
        param_name = "输出参数描述"
        system_prompt = f"根据上面提供的接口的相关信息，根据要求将**输出参数**的描述结构化。\n{require_prompt}\n直接返回接口输出描述的json："
    interface_infomation = f"一、接口名称: {interface_name.strip()}\n\n二、接口调用示例: {example.strip()}\n\n三、{param_name}: {param_description.strip()}"

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
        f"\nparameters:\n{json.dumps(obj=parameters, ensure_ascii=False, indent=2)}"
    )
    return set_property(
        driver=driver,
        database=database,
        interface_id=interface_id,
        property_name=property_name,
        property_value=parameters,
    )


def convert_interface_param_descriptions_to_struct(
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
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
    )
    with service_tool.driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""")
        interface_ids = [record["id"] for record in result]
        for interface_id in tqdm(interface_ids, desc="interface param descrition"):
            convert_interface_param_description_to_struct(
                driver=service_tool.driver,
                database=database,
                interface_id=interface_id,
                param_type="input",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )
            convert_interface_param_description_to_struct(
                driver=service_tool.driver,
                database=database,
                interface_id=interface_id,
                param_type="output",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )


if __name__ == "__main__":
    url = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    convert_interface_param_descriptions_to_struct(
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
