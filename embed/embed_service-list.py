import sys, os

sys.path.append(os.getcwd())
import numpy as np
from neo4j import GraphDatabase, Driver
from neo4j_haystack.document_stores import Neo4jDocumentStore
from utils.utils import openai_embedding
from collections import defaultdict
from tqdm import tqdm
from loguru import logger

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "service-cim-2025-12-16"

LABEL_TO_PROPERTIES_DICT = {
    "Interface": [
        "name",
        "llm_description",
        "standard_name",
        # "input_description",
        # "output_description",
    ],
    "Parameter": ["name", "chinese_name", "format", "description"],
    "Person": ["name"],
    "Service": ["name", "description", "standard_name", "standard_description"],
    "CIMClass": ["name", "description"],
    "CIMProperty": ["class", "property", "fullName"],
}

EMBEDDING_MODEL_NAME = "text-embedding-qwen3-embedding-8b"
BATCH_SIZE = 32


def connect_to_database(uri: str, user: str, password: str, database: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    logger.info(f"Connected to database: {database}")
    return driver


def fetch_all_nodes_by_label(driver, database_name):
    nodes_by_label = defaultdict(list)
    query = "MATCH (n) RETURN n, elementId(n) AS internal_id"
    with driver.session(database=database_name) as session:
        result = session.run(query)
        for record in result:
            node = record["n"]
            internal_id = record["internal_id"]
            label = list(node.labels)[0]
            properties = dict(node)
            properties["_neo4j_internal_id"] = internal_id
            nodes_by_label[label].append(properties)
    return nodes_by_label


def write_embeddings_to_db(driver, batch, database_name):
    query = """
    UNWIND $batch AS data
    MATCH (n)
    WHERE elementId(n) = data.id
    SET n.embedding = data.vector
    """
    params = {"batch": [{"id": id, "vector": vector.tolist()} for id, vector in batch]}
    with driver.session(database=database_name) as session:
        session.run(query, params)


from neo4j import GraphDatabase


def drop_index(driver, database, index_name):
    with driver.session(database=database) as session:
        indexes = session.run("SHOW INDEXES").data()
        existing_idx = [idx for idx in indexes if idx.get("name") == index_name]
        if not existing_idx:
            logger.info(
                f"[INFO] Index '{index_name}' does not exist. Nothing to delete."
            )
            return False
        try:
            session.run(f"DROP INDEX `{index_name}`")
            logger.info(f"[SUCCESS] Index '{index_name}' deleted.")
            return True
        except Exception as e:
            logger.info(f"[ERROR] Failed to drop index '{index_name}': {e}")
            return False


def generate_and_write_embeddings(
    nodes_by_label: dict,
    label_to_properties: dict,
    driver: Driver,
    database_name: str,
    embedding_base_url: str,
    model_name: str,
    batch_size: int,
):
    total_processed = 0
    for label, nodes_list in nodes_by_label.items():
        if label not in label_to_properties:
            continue

        props_to_use = label_to_properties[label]
        nodes_to_process = []

        for node_properties in nodes_list:
            internal_id = node_properties.pop("_neo4j_internal_id")
            texts_to_embed = [
                str(node_properties.get(prop, "")).strip() for prop in props_to_use
            ]
            combined_text = " ".join(filter(None, texts_to_embed))

            if combined_text:
                nodes_to_process.append((internal_id, combined_text))

        if not nodes_to_process:
            continue

        logger.info(f"\nExample text for label '{label}'")
        logger.info(nodes_to_process[0][1])

        for i in tqdm(
            range(0, len(nodes_to_process), batch_size),
            desc=f"Generating embeddings for label: {label}",
        ):
            batch = nodes_to_process[i : i + batch_size]
            ids = [item[0] for item in batch]
            texts = [item[1] for item in batch]

            embeddings = np.array(
                [
                    openai_embedding(
                        embedding_base_url=embedding_base_url,
                        model=model_name,
                        text=text,
                    )
                    for text in texts
                ]
            )

            batch_to_write = list(zip(ids, embeddings))
            write_embeddings_to_db(driver, batch_to_write, database_name)

        total_processed += len(nodes_to_process)

        drop_index(
            driver=driver, database=database_name, index_name=f"{label}-embedding"
        )
        Neo4jDocumentStore(
            url=NEO4J_URI,
            database=NEO4J_DATABASE,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index=f"{label}-embedding",
            node_label=label,
            progress_bar=True,
            embedding_dim=4096,
        )

    logger.info(
        f"\nFinished. Generated and wrote {total_processed} embeddings to the database."
    )


def main():
    embedding_base_url = os.getenv("EMBED_BASE_URL")
    driver = connect_to_database(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    try:
        nodes_by_label = fetch_all_nodes_by_label(driver, NEO4J_DATABASE)
        generate_and_write_embeddings(
            nodes_by_label=nodes_by_label,
            label_to_properties=LABEL_TO_PROPERTIES_DICT,
            driver=driver,
            database_name=NEO4J_DATABASE,
            embedding_base_url=embedding_base_url,
            model_name=EMBEDDING_MODEL_NAME,
            batch_size=BATCH_SIZE,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
