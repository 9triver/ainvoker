import sys, os

sys.path.append(os.getcwd())

from neo4j import GraphDatabase
from agent_system.actions.action import Action
from ..world_state import WorldState
from utils.utils import get_property
from json_repair import loads
from loguru import logger


class InterfaceAction(Action):
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

    def execute(self, state: WorldState, **kwargs) -> WorldState:
        interface_id = kwargs.get("interface_id")
        if not interface_id:
            return state

        interface_name = get_property(
            driver=self.driver,
            database=self.database,
            label="Interface",
            id=interface_id,
            property_name="name",
        )
        interface_description = get_property(
            driver=self.driver,
            database=self.database,
            label="Interface",
            id=interface_id,
            property_name="llm_description",
        )
        interface_output_params = loads(
            json_str=get_property(
                driver=self.driver,
                database=self.database,
                label="Interface",
                id=interface_id,
                property_name="output_params",
            )
        )

        return state.update(
            interfaces={interface_name: interface_description},
            parameters=interface_output_params,
        )
