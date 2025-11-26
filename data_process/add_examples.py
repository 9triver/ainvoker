import sys, os

sys.path.append(os.getcwd())
from utils.utils import read_excel, to_json
from neo4j import GraphDatabase, Driver
from tqdm import tqdm
from loguru import logger


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "service-list"


def extract_interfaces(api_file: str):
    dataframes = read_excel(path=api_file)
    interfaces = []
    for sheet_name, dataframe in dataframes.items():
        print(f"Sheet: {sheet_name}")
        service_list = to_json(df=dataframe)
        for service_category in service_list:
            for service in service_category["服务"]:
                for interface in service["接口"]:
                    interfaces.append(
                        {
                            "name": interface["接口名称"],
                            "standard_name": interface["标准化接口名称"],
                            "example": interface["调用示例（更新版）"],
                        }
                    )
    return interfaces


def insert_example_into_interface(
    driver: Driver,
    database: str,
    name: str,
    standard_name: str,
    example: str,
):
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (i:Interface {name: $name, standard_name: $standard_name})
            SET i.example = $example
            """,
            name=name,
            standard_name=standard_name,
            example=example,
        )
        summary = result.consume()
        if summary.counters.properties_set == 0:
            logger.warning(
                f"insert_example_into_interface failed: Interface(name={name}, standard_name={standard_name}) not found"
            )
    return


def main():
    interfaces = extract_interfaces(api_file="./data/服务清单.xlsx")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    for interface in tqdm(interfaces, desc="process interface"):
        interface_name = interface["name"]
        interface_standard_name = interface["standard_name"]
        interface_example = interface["example"]

        insert_example_into_interface(
            driver=driver,
            database=NEO4J_DATABASE,
            name=interface_name,
            standard_name=interface_standard_name,
            example=interface_example,
        )


if __name__ == "__main__":
    main()
