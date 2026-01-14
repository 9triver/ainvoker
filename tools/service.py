import os, sys, json

sys.path.append(os.getcwd())

from typing import List, Dict, Union, Any, Literal
from cn2an import an2cn

from agno.tools import Toolkit
from neo4j import GraphDatabase
from haystack import Document
from neo4j_haystack.document_stores import Neo4jDocumentStore
from utils.utils import openai_embedding, get_properties, get_property
from loguru import logger


class ServiceTools(Toolkit):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        embedding_base_url: str,
        embedding_model: str,
        enable_search_similar_cim_classes: bool = False,
        enable_search_similar_output_entities: bool = False,
        all: bool = False,
        **kwargs,
    ):
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.document_store = None

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or "neo4j"
        if self.user is None or self.password is None:
            raise ValueError("Username or password for Neo4j not provided")

        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.debug("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        tools: List[Any] = []
        if all or enable_search_similar_output_entities:
            tools.append(self.search_similar_output_entities)
        if all or enable_search_similar_cim_classes:
            tools.append(self.search_similar_cim_classes)
        super().__init__(name="service_tools", tools=tools, **kwargs)

    def _search_similar_nodes(
        self,
        text: str,
        node_label: str,
        top_k,
    ) -> List[Document]:
        if self.document_store is None:
            self.document_store = Neo4jDocumentStore(
                url=self.uri,
                database=self.database,
                username=self.user,
                password=self.password,
                embedding_dim=4096,
            )

        entity_embedding = openai_embedding(
            embedding_base_url=self.embedding_base_url,
            text=text,
            model=self.embedding_model,
        )
        self.document_store.node_label = node_label
        self.document_store.index = f"{node_label}-embedding"
        documents = self.document_store.query_by_embedding(
            query_embedding=entity_embedding, top_k=top_k
        )
        return documents

    def search_similar_output_entities(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        根据用户提供的查询文本，查找相似的输出业务实体和输出该实体接口id。

        Args:
            query (str): 用户的查询文本。
            top_k (int): 返回的相关业务实体数量，默认为 3。
        """
        entities = self._search_similar_nodes(
            text=query, node_label=f"OutputEntity", top_k=top_k
        )

        entity_contents = []
        for index, entity in enumerate(entities, 1):
            entity_id = entity.id
            entity_name = entity.meta["name"]
            entity_description = entity.meta["description"]
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (e {id: $entity_id})-[r:INPUT_ENTITY|OUTPUT_ENTITY]-(i:Interface)
                    RETURN DISTINCT i.id AS id, i.name AS name, i.llm_function_description AS description
                    """,
                    entity_id=entity_id,
                ).data()
                interface_id = result[0]["id"]
                interface_name = result[0]["name"]
                interface_description = result[0]["description"]

                entity_content = {
                    "序号": index,
                    "实体id": entity_id,
                    "实体名称": entity_name,
                    "实体描述": entity_description,
                    "相关接口id": interface_id,
                    "相关接口名称": interface_name,
                    "相关接口描述": interface_description,
                }
                entity_contents.append(entity_content)

        return json.dumps(obj=entity_contents, ensure_ascii=False, indent=2)
    
    def search_similar_cim_classes(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        根据用户提供的查询文本，查找相似的CIM类。

        Args:
            query (str): 用户的查询文本。
            top_k (int): 返回的相关业务实体数量，默认为 3。
        """
        cim_classes = self._search_similar_nodes(
            text=query, node_label=f"CIMClass", top_k=top_k
        )
        cim_class_contents = []
        for index, cim_class in enumerate(cim_classes, 1):
            cim_class_content = {
                "序号": index,
                **cim_class.meta,
            }
            cim_class_contents.append(cim_class_content)
        return json.dumps(obj=cim_class_contents, ensure_ascii=False, indent=2)
