from dataclasses import dataclass
from typing import Any, Dict, List, Optional, OrderedDict as OrderedDictType, Literal
from collections import OrderedDict
from itertools import zip_longest
import json


@dataclass
class WorldState:
    def __init__(
        self,
        mode: Literal["forward", "backward"],
        origin_question: str,
        interface_history: Optional[OrderedDictType[str, str]] = None,
        plan_history: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        current_goal: Optional[str] = None,
        final_goal: Optional[str] = None,
        goal_history: Optional[List[str]] = None,
    ):
        self._mode: Literal["forward", "backward"] = mode
        self._origin_question = origin_question
        self._interface_history = OrderedDict(interface_history or {})
        self._parameters = parameters or {}
        self._current_goal = current_goal
        self._final_goal = final_goal
        self._plan_history = plan_history or []
        self._goal_history = goal_history or []

    @property
    def mode(self) -> Literal["forward", "backward"]:
        return self._mode

    @property
    def origin_question(self) -> str:
        return self._origin_question

    @property
    def interface_history(self) -> OrderedDictType[str, str]:
        return self._interface_history

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def current_goal(self) -> Optional[str]:
        return self._current_goal

    @property
    def final_goal(self) -> Optional[str]:
        return self._final_goal

    @property
    def plan_history(self) -> List[str]:
        return self._plan_history

    @property
    def goal_history(self) -> List[str]:
        return self._goal_history

    @property
    def history(self) -> str:
        if self.mode == "forward":
            plan_iter = self.plan_history
            interface_iter = self.interface_history.items()
            goal_iter = self.goal_history
        elif self.mode == "backward":
            plan_iter = reversed(self.plan_history)
            interface_iter = reversed(list(self.interface_history.items()))
            goal_iter = reversed(self.goal_history)

        structured_steps = []
        for i, (plan, interface_item, goal) in enumerate(
            zip_longest(
                plan_iter,
                interface_iter,
                goal_iter,
                fillvalue="",
            ),
            1,
        ):
            interface_name = interface_item[0] if interface_item else None
            interface_desc = interface_item[1] if interface_item else None

            step_data = {
                "step_id": i,
                "step_goal": goal or None,
                "interface_call": (
                    {
                        "name": interface_name,
                        "description": interface_desc,
                    }
                    if interface_name
                    else None
                ),
                "plan": plan or None,
            }
            structured_steps.append(step_data)

        return structured_steps

    def __repr__(self) -> str:
        context = {
            "origin_question": self.origin_question,
            "final_goal": self.final_goal,
            "current_goal": self.current_goal,
            "history_steps": self.history,
        }
        return json.dumps(context, ensure_ascii=False, indent=2)

    def get_history_state(self) -> str:
        context = {
            "origin_question": self.origin_question,
            "history_steps": self.history,
        }
        return json.dumps(context, ensure_ascii=False, indent=2)

    def update(
        self,
        interface_history: Optional[OrderedDictType[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        current_goal: Optional[str] = None,
        plan: Optional[str] = None,
    ) -> "WorldState":
        # 接口
        new_interface_history = OrderedDict(self.interface_history)
        if interface_history:
            new_interface_history.update(interface_history)

        # 参数
        new_parameters = {**self.parameters, **(parameters or {})}

        # 目标链
        new_goal_history = self.goal_history.copy()
        new_current_goal = self.current_goal
        if current_goal:
            new_goal_history.append(self.current_goal)
            new_current_goal = current_goal

        # 计划
        new_plan_history = self.plan_history.copy()
        if plan:
            new_plan_history.append(plan)

        return WorldState(
            mode=self.mode,
            origin_question=self.origin_question,
            interface_history=new_interface_history,
            plan_history=new_plan_history,
            parameters=new_parameters,
            current_goal=new_current_goal,
            final_goal=self.final_goal,
            goal_history=new_goal_history,
        )

    def copy(self) -> "WorldState":
        return WorldState(
            mode=self.mode,
            origin_question=self.origin_question,
            plan_history=self.plan_history.copy(),
            interface_history=self.interface_history.copy(),
            parameters=self.parameters.copy(),
            current_goal=self.current_goal,
            final_goal=self.final_goal,
            goal_history=self.goal_history.copy(),
        )
