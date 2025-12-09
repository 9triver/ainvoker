import os, sys

sys.path.append(os.getcwd())

import numpy as np
from typing import List, Dict, Union, Any
from cn2an import an2cn

from agno.tools import Toolkit
from neo4j import GraphDatabase
from haystack import Document
from neo4j_haystack.document_stores import Neo4jDocumentStore
from utils.utils import openai_embedding
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
        enable_search_similar_interfaces: bool = False,
        enable_search_interface_by_name: bool = False,
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
        if all or enable_search_similar_interfaces:
            tools.append(self.find_similar_interfaces)
        if all or enable_search_interface_by_name:
            tools.append(self.find_interface_by_name)
        super().__init__(name="service_tools", tools=tools, **kwargs)

    def _find_similar_nodes(
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

    def _get_interface_details(
        self, interface_ids: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        if not interface_ids:
            return []
        if isinstance(interface_ids, str):
            interface_ids = [interface_ids]

        with self.driver.session(database=self.database) as session:
            query = """
            UNWIND $interface_ids AS interface_id
            MATCH (i:Interface {id: interface_id})
            OPTIONAL MATCH (in_param:Parameter)-[:INPUT_TO_INTERFACE]->(i)
            OPTIONAL MATCH (i)-[:OUTPUT_FROM_INTERFACE]->(out_param:Parameter)
            RETURN i.name AS name, i.standard_name AS standard_name,
                    i.description AS description, i.llm_description as llm_description,
                    i.input_description as input_description, i.output_description as output_description,
                    i.example as example,
                    COLLECT(DISTINCT {
                        name: in_param.name,
                        chinese_name: in_param.chinese_name,
                        field_type: in_param.field_type,
                        required: in_param.required,
                        description: in_param.description
                    }) AS input_parameters,
                    COLLECT(DISTINCT {
                        name: out_param.name,
                        chinese_name: out_param.chinese_name,
                        field_type: out_param.field_type,
                        description: out_param.description
                    }) AS output_parameters
            """

            result = session.run(query, interface_ids=interface_ids)

            interfaces = []
            for record in result:
                interface = {}
                for key in [
                    "name",
                    "description",
                    "llm_description",
                    "input_description",
                    "output_description",
                    "standard_name",
                    "example",
                    "input_parameters",
                    "output_parameters",
                ]:
                    if record.get(key) is not None:
                        interface[key] = record[key]
                interfaces.append(interface)
            return interfaces

    def find_similar_interfaces(
        self,
        task_description: str,
        top_k: int = 3,
    ) -> str:
        """
        根据用户提供的任务描述，查找内容相似的服务接口。

        Args:
            task_description (str): 用户的任务描述（用于匹配服务接口）。
            top_k (int): 返回的相关服务数量，默认为 3。
        """
        interfaces = self._find_similar_nodes(
            text=task_description, node_label="Interface", top_k=top_k
        )
        interface_ids = [interface.id for interface in interfaces]
        interface_details = self._get_interface_details(interface_ids=interface_ids)

        interface_contents = []
        for index, interface_detail in enumerate(interface_details):
            interface_id = interface_ids[index]
            name = interface_detail["name"]
            standard_name = interface_detail["standard_name"]
            llm_description = interface_detail["llm_description"]
            interface_content = (
                f"({an2cn(index+1)}) {name}/{standard_name}\n"
                + f"接口id: {interface_id}\n"
                + f"接口描述: {llm_description}"
            )
            interface_contents.append(interface_content)

        return "\n".join(interface_contents)

    def find_interface_by_name(self, name: str) -> str:
        """
        根据用户提供的接口名，查找目标服务接口的信息。
        Args:
            name (str): 接口名称。
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (i:Interface {name: $name})
                RETURN i.id as id
                """,
                name=name,
            )
            interface_id = result.data()[0]["id"]
            interface_detail = self._get_interface_details(interface_ids=interface_id)[
                0
            ]
            name = interface_detail["name"]
            standard_name = interface_detail["standard_name"]
            llm_description = interface_detail["llm_description"]
            interface_content = (
                f"({name}/{standard_name}\n"
                + f"接口id: {interface_id}\n"
                + f"接口描述: {llm_description}"
            )
            return interface_content
