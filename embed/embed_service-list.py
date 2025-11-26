NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "service-list"

LABEL_TO_PROPERTIES_DICT = {
    "Interface": ["name", "description", "standard_name", "input_description"],
    "Parameter": ["name", "chinese_name", "format", "description"],
    "Person": ["name"],
    "Service": ["name", "description", "standard_name", "standard_description"],
}

EMBEDDING_MODEL_NAME = "../model/Qwen/Qwen3-Embedding-4B"
BATCH_SIZE = 32

from neo4j import GraphDatabase, Driver
from neo4j_haystack.document_stores import Neo4jDocumentStore
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm


def connect_to_database(uri: str, user: str, password: str, database: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"Connected to database: {database}")
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


def generate_and_write_embeddings(
    nodes_by_label: dict,
    label_to_properties: dict,
    model: SentenceTransformer,
    driver: Driver,
    database_name: str,
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

        print(f"\nExample text for label '{label}'")
        print(nodes_to_process[0][1])

        for i in tqdm(
            range(0, len(nodes_to_process), batch_size),
            desc=f"Generating embeddings for label: {label}",
        ):
            batch = nodes_to_process[i : i + batch_size]
            ids = [item[0] for item in batch]
            texts = [item[1] for item in batch]

            embeddings = model.encode(texts, show_progress_bar=False)

            batch_to_write = list(zip(ids, embeddings))
            write_embeddings_to_db(driver, batch_to_write, database_name)

        total_processed += len(nodes_to_process)

        Neo4jDocumentStore(
            url=NEO4J_URI,
            database=NEO4J_DATABASE,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index=f"{label}-embedding",
            node_label=label,
            progress_bar=True,
            embedding_dim=model.get_sentence_embedding_dimension(),
        )

    print(
        f"\nFinished. Generated and wrote {total_processed} embeddings to the database."
    )


def main():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    driver = connect_to_database(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    try:
        nodes_by_label = fetch_all_nodes_by_label(driver, NEO4J_DATABASE)
        generate_and_write_embeddings(
            nodes_by_label,
            LABEL_TO_PROPERTIES_DICT,
            embedding_model,
            driver,
            NEO4J_DATABASE,
            BATCH_SIZE,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
