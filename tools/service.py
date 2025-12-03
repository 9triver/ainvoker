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
        tools.append(self.find_similar_cim_classes)
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

    def get_cim_class_details(
        self, class_names: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        查询 CIMClass 详细信息，包括 description、embedding、properties。

        支持传入节点的 `name`、节点属性 `id`，或 Neo4j 内部 id（通过 toString(id(c)) 比对）。

        Args:
            class_names (str | List[str]): 一个类名或类名列表。

        Returns:
            List[Dict[str, Any]]: CIMClass 详细信息列表。
        """
        if not class_names:
            return []
        if isinstance(class_names, str):
            class_names = [class_names]

        logger.debug(f"Getting CIM class details for: {class_names}")

        with self.driver.session(database=self.database) as session:
            # 匹配策略：优先按 c.name 或 c.id 属性匹配，兼容文档 id 为 Neo4j internal id 的情况
            query = """
            UNWIND $class_names AS class_name
            MATCH (c:CIMClass)
            WHERE c.name = class_name OR c.id = class_name OR elementId(c) = class_name
            OPTIONAL MATCH (c)-[:HAS_PROPERTY]->(p:CIMProperty)
            RETURN 
                c.name AS name,
                c.description AS description,
                c.embedding AS embedding,
                COLLECT(DISTINCT p.property) AS properties
            """

            result = session.run(query, class_names=class_names)

            classes = []
            for record in result:
                cls = {}
                for key in [
                    "name",
                    "description",
                    "embedding",
                    "properties",
                ]:
                    if record.get(key) is not None:
                        cls[key] = record[key]

                classes.append(cls)

            logger.debug(f"CIM class details found: {len(classes)}")
            return classes

    def find_similar_cim_classes(
        self,
        description: str,
        top_k: int = 3,
    ) -> str:
        """
        根据自然语言描述，查找最相似的 CIMClass 并返回详细信息文本。

        实现要点：
        - 使用 `_find_similar_nodes` 对 `CIMClass` 节点做向量检索；
        - 从返回的 `Document` 中稳健提取可用于查询 Neo4j 的标识（优先 `meta.name`，其次 `meta.id`，再 `content`，最后 `id`）；
        - 兼容文档 id 可能为 `internal id`、或带前缀的字符串（例如 `4:...`），将尝试同时提交原值和去掉冒号后的部分作为候选；
        - 调用 `get_cim_class_details` 获取完整节点信息并格式化输出。
        """

        # 1. 向量检索得到候选文档
        cim_classes = self._find_similar_nodes(
            text=description, node_label="CIMClass", top_k=top_k
        )

        # 2. 稳健提取用于匹配的 class_name 列表
        candidate_names: List[str] = []
        for doc in cim_classes:
            # haystack.Document 可能有 .meta (dict), .id (str), .content (str)
            name = None
            try:
                meta = getattr(doc, "meta", None)
                if isinstance(meta, dict):
                    name = meta.get("name") or meta.get("id") or meta.get("node_name")
            except Exception:
                name = None

            if not name:
                name = getattr(doc, "content", None) or getattr(doc, "id", None)

            if isinstance(name, str) and name:
                # 如果形如 '4:93e2...'，同时尝试原串与冒号前的 internal id
                candidate_names.append(name)
                if ":" in name:
                    candidate_names.append(name.split(":")[0])

        # 去重并移除空值
        candidate_names = [c for c in dict.fromkeys(candidate_names) if c]
        logger.debug(f"Extracted candidate class names for CIM lookup: {candidate_names}")

        if not candidate_names:
            return "（未检索到相关 CIMClass）"

        # 3. 获取详细信息
        details = self.get_cim_class_details(candidate_names)

        # 4. 拼接输出文本
        contents: List[str] = []
        for idx, cls in enumerate(details):
            name = cls.get("name", "(无名)")
            desc = cls.get("description", "(无描述)")
            embedding = cls.get("embedding", [])
            props = cls.get("properties", [])

            prop_lines = "\n".join([f"    - {p}" for p in props]) if props else "    （无属性）"

            content = f"""
({an2cn(idx+1)}) {name}
类描述：{desc}

属性列表：
{prop_lines}
"""
            contents.append(content)

        return "\n".join(contents)





