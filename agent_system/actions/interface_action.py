import sys, os

sys.path.append(os.getcwd())

from typing import List
from neo4j import GraphDatabase
from agent_system.world_state import WorldState
from utils.utils import flatten
from loguru import logger


class InterfaceAction:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
    ) -> None:
        super().__init__()
        self.database = database
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.debug("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        return

    def get_interface_by_interface_id(self, interface_id: str):
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                MATCH (i:Interface {id: $interface_id})
                RETURN i.id AS id, i.name AS name, i.llm_description AS llm_description
                """,
                interface_id=interface_id,
            ).single()
            if record is None:
                return None
            return {
                "id": record["id"],
                "name": record["name"],
                "llm_description": record["llm_description"],
            }

    def update_by_interface_ids(
        self, state: WorldState, interface_ids: List[str]
    ) -> WorldState:
        interface_ids = list(flatten(interface_ids))

        interface_calls = []
        new_obtained_entities: List[str] = state.obtained_entities
        new_required_entities: List[str] = []
        for interface_id in interface_ids:
            interface_info = self.get_interface_by_interface_id(
                interface_id=interface_id
            )
            if not interface_info:
                continue

            interface_calls.append(
                {
                    "id": interface_info["id"],
                    "name": interface_info["name"],
                    "description": interface_info["llm_description"],
                }
            )

        return state.update(
            interface_history=interface_calls,
            obtained_entities=new_obtained_entities,
            required_entities=new_required_entities,
        )

    @staticmethod
    def is_entity_id(entity_str: str):
        return entity_str.startswith("Entity_")
