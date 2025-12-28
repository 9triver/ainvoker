from typing import Dict, Optional, List
import json


class WorldState:
    def __init__(
        self,
        origin_question: str,
        interface_history: List[Dict[str, str]] = None,
        obtained_entities: List[str] = None,
        required_entities: List[str] = None,
    ):
        self._origin_question = origin_question
        self._interface_history = interface_history or []
        self._obtained_entities = obtained_entities or []
        self._required_entities = required_entities or []

    @property
    def origin_question(self) -> str:
        return self._origin_question

    @property
    def interface_history(self) -> List[Dict[str, str]]:
        return self._interface_history

    @property
    def obtained_entities(self) -> List[str]:
        return self._obtained_entities

    @property
    def required_entities(self) -> List[str]:
        return self._required_entities

    @property
    def interface_calls(self) -> str:
        interface_calls = []
        for i, interface_info in enumerate(self.interface_history, 1):
            interface_call = {
                "number": i,
                "id": interface_info["id"],
                "name": interface_info["name"],
                "description": interface_info["description"],
            }
            interface_calls.append(interface_call)

        return interface_calls

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorldState):
            return False
        return (
            self.origin_question == other.origin_question
            and self.interface_history == other.interface_history
            and self.obtained_entities == other.obtained_entities
            and self.required_entities == other.required_entities
        )

    def __repr__(self) -> str:
        context = {
            "origin_question": self.origin_question,
            "required_entities": self.required_entities,
            "candidate_interfaces": self.interface_history,
        }
        return json.dumps(context, ensure_ascii=False, indent=2)

    def update(
        self,
        interface_history: Optional[List[Dict[str, str]]] = None,
        obtained_entities: Optional[List[str]] = None,
        required_entities: Optional[List[str]] = None,
    ) -> "WorldState":
        # 接口
        new_interface_history = self.interface_history.copy()
        if interface_history:
            interface_ids = set([i["id"] for i in new_interface_history])
            for interface_info in interface_history:
                if interface_info["id"] in interface_ids:
                    continue
                new_interface_history.append(interface_info)

        # 实体
        new_obtained_entities = self.obtained_entities
        if obtained_entities:
            new_obtained_entities = obtained_entities
        new_required_entities = self.required_entities
        if obtained_entities:
            new_required_entities = required_entities

        return WorldState(
            origin_question=self.origin_question,
            interface_history=new_interface_history,
            obtained_entities=new_obtained_entities,
            required_entities=new_required_entities,
        )

    def copy(self) -> "WorldState":
        return WorldState(
            origin_question=self.origin_question,
            interface_history=self.interface_history.copy(),
            obtained_entities=self.obtained_entities.copy(),
            required_entities=self.required_entities.copy(),
        )
