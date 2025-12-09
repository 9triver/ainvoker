from dataclasses import dataclass
from typing import Any, Dict, Optional, List, OrderedDict as OrderedDictType
from itertools import zip_longest
from collections import OrderedDict
import json


@dataclass
class WorldState:
    def __init__(
        self,
        origin_question: str,
        interfaces: Optional[OrderedDictType[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parameter_history: Optional[List[Dict[str, Any]]] = None,
        current_goal: Optional[str] = None,
        goal_history: Optional[List[str]] = None,
        plans: Optional[List[str]] = None,
    ):
        self._origin_question = origin_question
        self._interfaces = OrderedDict(interfaces or {})
        self._parameters = parameters or {}
        self._parameter_history = parameter_history or []
        self._current_goal = current_goal
        self._plans = plans or []
        self._goal_history = goal_history or []

    @property
    def parameter_history(self) -> List[Dict[str, Any]]:
        return self._parameter_history

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def current_goal(self) -> str:
        return self._current_goal

    @property
    def origin_question(self) -> str:
        return self._origin_question

    @property
    def interfaces(self) -> OrderedDictType[str, str]:
        return self._interfaces

    @property
    def plans(self) -> List[str]:
        return self._plans

    @property
    def goal_history(self) -> List[str]:
        return self._goal_history

    @property
    def history(self) -> str:
        steps = []
        for i, (plan, interface_item, goal) in enumerate(
            zip_longest(
                self._plans, self._interfaces.items(), self._goal_history, fillvalue=""
            ),
            1,
        ):
            interface_name = interface_item[0] if interface_item else ""
            interface_desc = interface_item[1] if interface_item else ""

            steps.append(f"  步骤 {i}:")
            steps.append(f"    目标: {goal}")
            steps.append(f"    接口: {interface_name}: {interface_desc}")
            steps.append(f"    计划: {plan}")

        return "\n".join(steps) if steps else "  (无历史记录)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interfaces": dict(self._interfaces),
            "plans": self.plans,
            "origin_question": self.origin_question,
            "goal_history": self.goal_history,
            "current_goal": self.current_goal,
            "parameter_history": self._parameter_history,
            "parameters": self.parameters,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, WorldState):
            return NotImplemented
        return (
            self._parameters == other._parameters
            and self._current_goal == other._current_goal
            and self._origin_question == other._origin_question
            and dict(self._interfaces) == dict(other._interfaces)
            and self._plans == other._plans
            and self._goal_history == other._goal_history
            and self._parameter_history == other._parameter_history
        )

    def __hash__(self) -> int:
        params_hash = tuple(
            sorted(
                (key, tuple(value) if isinstance(value, list) else value)
                for key, value in self._parameters.items()
            )
        )
        interfaces_hash = tuple(self._interfaces.items())
        plans_hash = tuple(self._plans)
        goal_history_hash = tuple(self._goal_history)
        param_history_hash = tuple(
            tuple(
                sorted(
                    (key, tuple(value) if isinstance(value, list) else value)
                    for key, value in param_snapshot.items()
                )
            )
            for param_snapshot in self._parameter_history
        )

        return hash(
            (
                params_hash,
                self._current_goal,
                self._origin_question,
                interfaces_hash,
                plans_hash,
                goal_history_hash,
                param_history_hash,
            )
        )

    def __repr__(self) -> str:
        params_lines = []
        for key, value in self._parameters.items():
            if isinstance(value, (dict, list)):
                try:
                    value_str = json.dumps(value, ensure_ascii=False, indent=2)
                except:
                    value_str = str(value)
            else:
                value_str = str(value)
            params_lines.append(f"  {key}: {value_str}")
        params_str = "\n".join(params_lines)

        history_str = self.history

        return (
            f"世界状态:\n\n"
            f"历史执行步骤 ({len(self._plans)}个):\n"
            f"{history_str}\n\n"
            f"当前可用参数 ({len(self._parameters)}个):\n"
            f"{params_str}\n\n"
            f"当前目标: {self._current_goal}\n"
            f"初始问题: {self._origin_question}\n"
        )

    def get_history_state(self) -> str:
        history_str = self.history
        return (
            f"世界状态:\n"
            f"历史执行步骤 ({len(self._plans)}个):\n"
            f"{history_str}\n\n"
            f"初始问题: {self._origin_question}\n"
        )

    def copy(self) -> "WorldState":
        return WorldState(
            origin_question=self._origin_question,
            parameters=self.parameters,
            current_goal=self.current_goal,
            interfaces=self.interfaces,
            plans=self.plans,
            goal_history=self.goal_history,
            parameter_history=self._parameter_history.copy(),
        )

    def update(
        self,
        interfaces: Optional[OrderedDictType[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        current_goal: Optional[str] = None,
        plan: Optional[str] = None,
    ) -> "WorldState":
        new_interfaces = OrderedDict(self._interfaces)
        if interfaces:
            new_interfaces.update(interfaces)

        new_parameters = {**self._parameters, **(parameters or {})}
        new_parameter_history = self._parameter_history.copy()
        new_parameter_history.append(new_parameters.copy())

        new_goal_history = self._goal_history.copy()
        new_current_goal = self._current_goal
        if current_goal:
            new_current_goal = current_goal
            if self._current_goal:
                new_goal_history.append(self._current_goal)

        new_plans = self._plans.copy()
        if plan:
            new_plans.append(plan)

        return WorldState(
            origin_question=self._origin_question,
            parameters=new_parameters,
            current_goal=new_current_goal,
            interfaces=new_interfaces,
            plans=new_plans,
            goal_history=new_goal_history,
            parameter_history=new_parameter_history,
        )

    def roll_back(self, steps: int) -> "WorldState":
        if steps <= 0:
            return self.copy()
        target_step_count = max(len(self._plans) - steps, 0)

        target_plans = self._plans[:target_step_count]
        target_goal_history = self._goal_history[:target_step_count]
        target_interfaces = OrderedDict(
            list(self._interfaces.items())[:target_step_count]
        )
        target_parameters = self._parameter_history[target_step_count].copy()
        target_current_goal = (
            self._goal_history[target_step_count - 1] if target_step_count > 0 else None
        )
        target_parameter_history = self._parameter_history[: target_step_count + 1]

        return WorldState(
            origin_question=self._origin_question,
            parameters=target_parameters,
            current_goal=target_current_goal,
            interfaces=target_interfaces,
            plans=target_plans,
            goal_history=target_goal_history,
            parameter_history=target_parameter_history,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldState":
        interfaces = data.get("interfaces", {})
        if not isinstance(interfaces, OrderedDict):
            interfaces = OrderedDict(interfaces)

        parameter_history = data.get("parameter_history")
        if parameter_history is None:
            parameter_history = [data.get("parameters", {}).copy()]

        return cls(
            origin_question=data.get("origin_question", ""),
            parameters=data.get("parameters", {}),
            current_goal=data.get("current_goal", ""),
            interfaces=interfaces,
            plans=data.get("plans", []),
            goal_history=data.get("goal_history", []),
            parameter_history=parameter_history,
        )
