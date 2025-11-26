import sys, os, time

sys.path.append(os.getcwd())
from tqdm import tqdm
from tools.service import ServiceTools
from utils.utils import ask_llm
from loguru import logger


def has_property(driver, database: str, interface_id: str, property_name: str) -> bool:
    query = f"""
    MATCH (n:Interface {{id: $interface_id}})
    RETURN (n.{property_name} IS NOT NULL) AS has_prop
    """
    with driver.session(database=database) as session:
        result = session.run(query, interface_id=interface_id).single()
        return result["has_prop"] if result else False


def set_property(
    driver, database: str, interface_id: str, property_name: str, property_value
):
    query = f"""
    MATCH (n:Interface {{id: $interface_id}})
    SET n.{property_name} = $property_value
    RETURN n.{property_name} AS new_value
    """
    with driver.session(database=database) as session:
        result = session.run(
            query, interface_id=interface_id, property_value=property_value
        ).single()
        return result["new_value"] if result else None


def add_interface_param_description(
    driver,
    database: str,
    interface_id: str,
    interface_detail: dict,
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
        driver,
        database=database,
        interface_id=interface_id,
        property_name=property_name,
    ):
        return

    input_interface_detail = {}
    for key in [
        "name",
        "description",
        "standard_name",
        "example",
        detail_key,
    ]:
        if interface_detail.get(key) is not None:
            input_interface_detail[key] = interface_detail[key]
    user_prompt = "{input_interface_detail}\n{prompt}".format(
        input_interface_detail=input_interface_detail,
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
        interface_id=interface_id,
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
    service_tool = ServiceTools(
        uri=uri,
        user=user,
        password=password,
        database=database,
    )
    with service_tool.driver.session(database=database) as session:
        result = session.run("""MATCH (n:Interface) RETURN n.id AS id""")
        interface_ids = [record["id"] for record in result]
        for interface_id in tqdm(interface_ids, desc="interface param descrition"):
            interface_detail = service_tool._get_interface_details(
                interface_ids=interface_id
            )[0]
            add_interface_param_description(
                driver=service_tool.driver,
                database=database,
                interface_id=interface_id,
                interface_detail=interface_detail,
                param_type="input",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )
            add_interface_param_description(
                driver=service_tool.driver,
                database=database,
                interface_id=interface_id,
                interface_detail=interface_detail,
                param_type="output",
                api_key_name=api_key_name,
                base_url=base_url,
                model=model,
            )


if __name__ == "__main__":
    url = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    add_interface_param_descriptions(
        uri=url,
        user=user,
        password=password,
        database="service-list",
        # api_key_name="MODELSCOPE_API_KEY",
        # base_url="https://api-inference.modelscope.cn/v1",
        api_key_name="CHATGLM_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4.5-flash",
    )
