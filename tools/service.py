import os, sys

sys.path.append(os.getcwd())

import numpy as np
from typing import List, Dict, Union, Any, Optional
from cn2an import an2cn

from agno.tools import Toolkit
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from haystack import Document
from neo4j_haystack.document_stores import Neo4jDocumentStore
from utils.utils import lm_studio_embedding
from loguru import logger


class ServiceTools(Toolkit):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        base_url: str,
        embedding_model: str,
        enable_search_interfaces: bool = False,
        all: bool = False,
        **kwargs,
    ):
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.document_store = None

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or "neo4j"
        if self.user is None or self.password is None:
            raise ValueError("Username or password for Neo4j not provided")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))  # type: ignore
            self.driver.verify_connectivity()
            logger.debug("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        tools: List[Any] = []
        if all or enable_search_interfaces:
            tools.append(self.find_similar_interfaces)
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

        entity_embedding = lm_studio_embedding(
            base_url=self.base_url, text=text, model=self.embedding_model
        )
        self.document_store.node_label = node_label
        self.document_store.index = f"{node_label}-embedding"
        documents = self.document_store.query_by_embedding(
            query_embedding=entity_embedding, top_k=top_k
        )
        return documents

    def _find_interfaces_by_parameters(
        self, param_ids: List[str]
    ) -> List[Dict[str, Any]]:
        if not param_ids:
            return []

        with self.driver.session(database=self.database) as session:
            query = """
            UNWIND $param_ids AS param_id
            MATCH (p:Parameter {id: param_id})-[:INPUT_TO_INTERFACE]->(s:Interface)
            RETURN DISTINCT s.id AS id, s.name AS name, 
                    s.description AS description, s.standard_name AS standard_name,
                    s.embedding AS embedding
            """

            result = session.run(query, param_ids=param_ids)

            interfaces = []
            for record in result:
                interfaces.append(
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "description": record["description"],
                        "standard_name": record["standard_name"],
                        "embedding": record["embedding"],
                    }
                )

            return interfaces

    def _rank_interfaces_by_task_similarity(
        self, interfaces: List[Dict[str, Any]], task_description: str
    ) -> List[Dict[str, Any]]:
        if not interfaces:
            return []
        task_embedding = np.array(
            lm_studio_embedding(
                base_url=self.base_url,
                model=self.embedding_model,
                text=task_description,
            )
        ).reshape(1, -1)
        interface_ids = [s["id"] for s in interfaces]

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (n:Interface) 
            WHERE n.id IN $interface_ids
            RETURN n.id AS id, n.embedding AS embedding
            """
            result = session.run(query, interface_ids=interface_ids)
            embedding_map = {
                record["id"]: np.array(record["embedding"])
                for record in result
                if record["embedding"] is not None and len(record["embedding"]) > 0
            }

        for interface in interfaces:
            interface_id = interface["id"]
            interface_embedding = embedding_map.get(interface_id)
            if interface_embedding is None:
                interface["task_similarity"] = 0.0
                continue
            interface_embedding = interface_embedding.reshape(1, -1)
            interface["task_similarity"] = float(
                cosine_similarity(task_embedding, interface_embedding)[0][0]
            )

        interfaces.sort(key=lambda x: x["task_similarity"], reverse=True)
        return interfaces

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
            MATCH (s:Interface {id: interface_id})
            OPTIONAL MATCH (in_param:Parameter)-[:INPUT_TO_INTERFACE]->(s)
            OPTIONAL MATCH (s)-[:OUTPUT_FROM_INTERFACE]->(out_param:Parameter)
            RETURN s.name AS name, s.standard_name AS standard_name, s.description AS description,
                    s.input_description as input_description, s.output_description as output_description,
                    s.example as example,
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

    def find_interfaces_by_entity_and_task(
        self,
        entity_text: str,
        task_description: str,
        top_k_params: int = 5,
        top_k_interfaces: int = 5,
    ) -> Dict[str, Any]:
        """
        根据用户提供的参数实体和任务描述，查找匹配的服务接口。

        Args:
            entity_text (str): 用户提到的实体文本（用于匹配参数）。
            task_description (str): 用户的任务描述（用于匹配服务接口）。
            top_k_params (int): 用于匹配的相似参数数量，默认为 5。
            top_k_interfaces (int): 返回的最相关服务数量，默认为 5。
        """
        similar_params = self._find_similar_nodes(
            text=entity_text, node_label="Parameter", top_k=top_k_params
        )

        param_ids = [param.id for param in similar_params]
        interfaces = self._find_interfaces_by_parameters(param_ids=param_ids)

        ranked_interfaces = self._rank_interfaces_by_task_similarity(
            interfaces=interfaces, task_description=task_description
        )[:top_k_interfaces]

        top_interface_ids = [
            interface["id"] for interface in ranked_interfaces[:top_k_interfaces]
        ]
        interface_details = self._get_interface_details(interface_ids=top_interface_ids)

        return interface_details

    def find_similar_interfaces(
        self,
        task_description: str,
        top_k: int = 3,
    ) -> str:
        """
        根据用户提供的任务描述，查找内容相似的服务接口。

        Args:
            task_description (str): 用户的任务描述（用于匹配服务接口）。
            top_k (int): 返回的最相关服务数量，默认为 3。
        """
        interfaces = self._find_similar_nodes(
            text=task_description, node_label="Interface", top_k=top_k
        )
        interface_ids = [interface.id for interface in interfaces]
        interface_details = self._get_interface_details(interface_ids=interface_ids)

        interface_contents = []
        for index, interface_detail in enumerate(interface_details):
            name = interface_detail["name"]
            standard_name = interface_detail["standard_name"]
            description = interface_detail["description"]
            input_description = interface_detail["input_description"]
            output_description = interface_detail["output_description"]
            interface_content = f"""\
({an2cn(index+1)}) {name}/{standard_name}
接口描述：{description}
输入描述：{input_description}
输出描述：{output_description}
"""
            interface_contents.append(interface_content)

        return "\n".join(interface_contents)
