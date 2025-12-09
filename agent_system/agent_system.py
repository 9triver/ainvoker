import sys, os

sys.path.append(os.getcwd())

from agno.models.base import Model
from agno.agent import Agent, RunOutput
from agno.utils.pprint import pprint_run_response
from pydantic import BaseModel, Field
from json_repair import loads

from tools.service import ServiceTools
from .world_state import WorldState
from .actions import InterfaceAction


class PlanStep(BaseModel):
    interface_id: str = Field(..., description="完成当前目标所要调用的接口id")
    plan: str = Field(..., description="完成当前目标需要如何调用接口")
    next_goal: str = Field(
        ...,
        description="如果没有解决原始问题，填入下一步要完成的目标; 如果解决了，设为空字符串",
    )


class CheckStep(BaseModel):
    reasonable: bool = Field(..., description="当前目标是否可行")
    hint: str = Field(
        ..., description="具体完成可行目标的提示信息。或更换可行目标的提示信息。"
    )


class AgentSystem:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ) -> None:
        self.planner = AgentSystem.init_planner(
            uri=uri,
            user=user,
            password=password,
            database=database,
            model=model,
            embedding_base_url=embedding_base_url,
        )
        self.checker = AgentSystem.init_checker(
            uri=uri,
            user=user,
            password=password,
            database=database,
            model=model,
            embedding_base_url=embedding_base_url,
        )

        self.interface_action = InterfaceAction(
            uri=uri, user=user, password=password, database=database
        )
        self.summarizer = AgentSystem.init_summarizer(
            uri=uri,
            user=user,
            password=password,
            database=database,
            model=model,
            embedding_base_url=embedding_base_url,
        )

    def response(self, question: str, max_step: int = 10) -> str:
        plan_step = PlanStep(interface_id="", plan="", next_goal="start step")
        world_state = WorldState(
            origin_question=question,
            current_goal="提取出原始问题中可用信息作为参数，思考第一步的目标是什么并完成它",
        )

        hint = ""
        step_num = 0
        while plan_step.next_goal and step_num < max_step:
            # plan
            input_str = str(world_state)
            if hint:
                input_str += f"(提示: {hint})\n"
            response: RunOutput = self.planner.run(
                input=input_str,
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            # str -> json -> PlanStep
            response_obj = loads(response.content)
            if isinstance(response_obj, list):
                response_obj = response_obj[-1]
            plan_step = PlanStep(**loads(response.content))

            # update
            world_state = world_state.update(
                current_goal=plan_step.next_goal, plan=plan_step.plan
            )
            world_state = self.interface_action.execute(
                state=world_state, interface_id=plan_step.interface_id
            )

            # check
            response: RunOutput = self.checker.run(
                input=str(world_state),
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            # str -> json -> CheckStep
            response_obj = loads(response.content)
            if isinstance(response_obj, list):
                response_obj = response_obj[-1]
            check_step = CheckStep(**loads(response.content))
            if not check_step.reasonable:
                world_state = world_state.roll_back(steps=1)
            hint = check_step.hint
            # step++
            step_num += 1

        response: RunOutput = self.summarizer.run(
            input=world_state.get_history_state(),
            debug_mode=True,
        )
        pprint_run_response(response, markdown=True)
        return response.content

    @staticmethod
    def init_planner(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        return Agent(
            name="Plan Agent",
            model=model,
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="text-embedding-qwen3-embedding-8b",
                    enable_search_similar_interfaces=True,
                )
            ],
            role="完成当前目标并制定下个目标",
            instructions=[
                "1. 结合用户原始问题、当前状态、可用参数、当前目标写一个查询文本, 并使用工具搜索 **直接/辅助** 完成目标的服务接口。",
                "2. 返回的服务接口不符合预期时，多加思考并且多调用工具。按要求填写接口id和计划描述。",
                "3. 如果没有解决原始问题，按要求填写下一步要完成的目标。",
                "4. 返回的json格式如下：{'interface_id': ..., 'plan': ..., 'next_goal': ...}",
                "4.1 interface_id(str): 直接/辅助完成当前目标所要调用的接口id",
                "4.2 plan(str): 直接/辅助完成当前目标需要如何调用接口, 用户需要额外提供哪些参数",
                "4.3 next_goal(str):如果没有解决原始问题，填入下一步要完成的目标; 如果解决了，设为空字符串",
                "5. 注意：接口无法调用，只需要提供接口调用的计划即可。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_summarizer(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        return Agent(
            name="Summarizer Agent",
            model=model,
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="text-embedding-qwen3-embedding-8b",
                    enable_search_interface_by_name=True,
                )
            ],
            role="整理信息和收集数据后回答用户原始问题",
            instructions=[
                "1. 根据状态信息中调用的接口和对应的计划，梳理相关数据。",
                "2. 可以根据接口名称调用工具，来查看接口的详细信息。",
                "3. 返回给我完善的计划，来回答用户的原始问题。",
                "4. 注意：接口无法调用，只需要提供接口调用的计划即可。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_checker(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        return Agent(
            name="Check Agent",
            model=model,
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="text-embedding-qwen3-embedding-8b",
                    enable_search_similar_interfaces=True,
                )
            ],
            role="检查当前的目标和计划是否可行",
            instructions=[
                "1. 结合用户原始问题、当前状态、可用参数来判断最近的一个历史执行步骤是否合理。",
                "2. 你可以使用工具去检索额外的服务接口信息，来验证最近一个历史执行步骤的可行性。",
                "3. 按要求填写可行性和提示信息。返回的json格式如下：{'reasonable': ..., 'hint': ...}",
                "3.1 reasonable(bool): 最近的一步历史执行步骤是否合理",
                "3.2 hint(str): 如果可行，接下来如何完成当前目标的提示信息。如果不可行，重新执行上一步的目标提示信息。",
                "4. 注意：接口无法调用，只需要提供接口调用的计划即可。缺少的参数可以暂时在计划中要求用户后续提供。",
            ],
            markdown=True,
        )
