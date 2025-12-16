import sys, os

sys.path.append(os.getcwd())

from typing import Literal
from agno.models.base import Model
from agno.agent import Agent, RunOutput
from agno.utils.pprint import pprint_run_response
from pydantic import BaseModel, Field
from json_repair import loads

from tools.service import ServiceTools
from .world_state import WorldState
from .actions import InterfaceAction


class ForwardPlanStep(BaseModel):
    interface_id: str = Field(..., description="完成当前目标所要调用的接口id")
    plan: str = Field(..., description="完成当前目标需要如何调用接口")
    next_goal: str = Field(
        ...,
        description="如果没有解决原始问题，填入下一步要完成的目标; 如果解决了，设为空字符串",
    )


class ForwardCheckStep(BaseModel):
    reasonable: bool = Field(..., description="当前目标是否可行")
    hint: str = Field(
        ..., description="具体完成可行目标的提示信息。或更换可行目标的提示信息。"
    )


class BackwardPlanStep(BaseModel):
    interface_id: str = Field(..., description="能产出当前目标所需结果的接口id")
    plan: str = Field(..., description="如何利用该接口的输出来满足当前目标")
    previous_goal: str = Field(
        ...,
        description="为了调用该接口，需要满足的前置条件（即上一步需要完成的目标）。如果用户原始问题已满足所有条件，设为空字符串",
    )


class BackwardCheckStep(BaseModel):
    reasonable: bool = Field(..., description="当前规划的前置步骤是否合理")
    hint: str = Field(
        ...,
        description="如果合理，给出继续推导前置条件的提示；如果不合理，给出重新规划的提示。",
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
        mode: Literal["forward", "backward"],
    ) -> None:
        self.mode: Literal["forward", "backward"] = mode
        self.interface_action = InterfaceAction(
            uri=uri, user=user, password=password, database=database
        )
        if mode == "forward":
            self.goaler = AgentSystem.init_goal_agent(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
            self.planner = AgentSystem.init_forward_planner(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
            self.checker = AgentSystem.init_forward_checker(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
            self.summarizer = AgentSystem.init_forward_summarizer(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
        elif mode == "backward":
            self.goaler = AgentSystem.init_goal_agent(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
            self.planner = AgentSystem.init_backward_planner(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
            self.checker = AgentSystem.init_backward_checker(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
            self.summarizer = AgentSystem.init_backward_summarizer(
                uri=uri,
                user=user,
                password=password,
                database=database,
                model=model,
                embedding_base_url=embedding_base_url,
            )
        return

    def response(self, question: str, max_step: int = 10) -> str:
        if self.mode == "forward":
            return self.response_forward(question=question, max_step=max_step)
        elif self.mode == "backward":
            return self.response_backward(question=question, max_step=max_step)

    def response_forward(self, question: str, max_step: int) -> str:
        # 1. generate final_goal
        response: RunOutput = self.goaler.run(
            input=f"用户问题：{question}\n请输出规划的最终目标。",
            debug_mode=True,
        )
        pprint_run_response(response, markdown=True)
        response_obj = loads(response.content)
        if isinstance(response_obj, list):
            response_obj = response_obj[-1]
        final_goal = response_obj["final_goal"]

        # 2. init plan_step and world_state
        plan_step = ForwardPlanStep(interface_id="", plan="", next_goal="start step")
        current_world_state = WorldState(
            mode="forward",
            origin_question=question,
            current_goal="思考第一步的目标是什么并完成它",
            final_goal=final_goal,
        )
        plan_world_state = current_world_state.copy()

        hint = "用户原始问题中包含一些实体, 这些实体可以拿来作为查询信息"
        step_num = 0
        while plan_step.next_goal and step_num < max_step:
            # forward plan
            input_str = str(plan_world_state)
            if hint:
                input_str += f"(提示: {hint})\n"
            response: RunOutput = self.planner.run(
                input=input_str,
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            response_obj = loads(response.content)
            if isinstance(response_obj, list):
                response_obj = response_obj[-1]
            plan_step = ForwardPlanStep(**loads(response.content))

            # update
            current_world_state = current_world_state.update(
                current_goal=plan_step.next_goal, plan=plan_step.plan
            )
            current_world_state = self.interface_action.execute(
                state=current_world_state, interface_id=plan_step.interface_id
            )

            # forward check
            response: RunOutput = self.checker.run(
                input=str(current_world_state),
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            response_obj = loads(response.content)
            if isinstance(response_obj, list):
                response_obj = response_obj[-1]
            check_step = ForwardCheckStep(**loads(response.content))
            if check_step.reasonable:
                plan_world_state = current_world_state.copy()
            else:
                plan_world_state, current_world_state = (
                    current_world_state.copy(),
                    plan_world_state.copy(),
                )
            hint = check_step.hint
            # step++
            step_num += 1

        # summarize
        response: RunOutput = self.summarizer.run(
            input=current_world_state.get_history_state(),
            debug_mode=True,
        )
        pprint_run_response(response, markdown=True)
        return response.content

    def response_backward(self, question: str, max_step: int) -> str:
        # 1. generate final_goal
        response: RunOutput = self.goaler.run(
            input=f"用户问题：{question}\n请输出规划的最终目标。",
            debug_mode=True,
        )
        pprint_run_response(response, markdown=True)
        response_obj = loads(response.content)
        if isinstance(response_obj, list):
            response_obj = response_obj[-1]
        final_goal = response_obj["final_goal"]

        # 2. init plan_step and world_state
        plan_step = BackwardPlanStep(
            interface_id="", plan="", previous_goal="start planning"
        )
        current_world_state = WorldState(
            mode="backward",
            origin_question=question,
            current_goal=final_goal,
            final_goal=final_goal,
        )
        plan_world_state = current_world_state.copy()

        hint = (
            "请根据当前目标，反向查找能产生该结果的接口，并确定该接口的输入作为前置目标"
        )
        step_num = 0
        while plan_step.previous_goal and step_num < max_step:
            # backward plan
            input_str = str(plan_world_state)
            if hint:
                input_str += f"(提示: {hint})\n"

            response: RunOutput = self.planner.run(
                input=input_str,
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            response_obj = loads(response.content)
            if isinstance(response_obj, list):
                response_obj = response_obj[-1]
            plan_step = BackwardPlanStep(**response_obj)

            current_world_state = current_world_state.update(
                current_goal=plan_step.previous_goal, plan=plan_step.plan
            )
            current_world_state = self.interface_action.execute(
                state=current_world_state, interface_id=plan_step.interface_id
            )

            # backward check
            response: RunOutput = self.checker.run(
                input=str(current_world_state),
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            response_obj = loads(response.content)
            if isinstance(response_obj, list):
                response_obj = response_obj[-1]
            check_step = BackwardCheckStep(**response_obj)
            if check_step.reasonable:
                plan_world_state = current_world_state.copy()
            else:
                plan_world_state, current_world_state = (
                    current_world_state.copy(),
                    plan_world_state.copy(),
                )
            hint = check_step.hint
            step_num += 1

        # summarize
        response: RunOutput = self.summarizer.run(
            input=current_world_state.get_history_state(),
            debug_mode=True,
        )
        pprint_run_response(response, markdown=True)
        return response.content

    @staticmethod
    def init_goal_agent(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        return Agent(
            name="Backward Goal Agent",
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
            role="生成反向规划的最终目标",
            instructions=[
                "1. 根据用户原始问题，给出接下来规划路径的最终目标（final_goal）。",
                "2. final_goal 必须是一个明确的、高层次的、可执行/可推导的目标。",
                "3. 输出 JSON 格式：{'final_goal': ...}",
                "4. 禁止返回空字符串。可以使用工具查询相似接口",
            ],
            markdown=True,
        )

    @staticmethod
    def init_forward_planner(
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
                "1. 分析相关信息, 使用工具搜索 **直接/辅助** 完成目标的服务接口。",
                "2. 返回的服务接口不符合预期时，多加思考并且多调用工具。按要求填写接口id和计划描述。",
                "3. 如果最终目标(final_goal)尚未完成，按要求填写下一步要完成的目标。",
                "4. JSON格式: {'interface_id': ..., 'plan': ..., 'next_goal': ...}",
                "4.1 interface_id(str): 直接/辅助完成当前目标(current_goal)所要调用的接口id",
                "4.2 plan(str): 直接/辅助完成当前目标(current_goal)需要如何调用接口, 用户需要额外提供哪些参数",
                "4.3 next_goal(str):如果最终目标(final_goal)尚未完成，填入下一步要完成的目标; 如果解决了，设为空字符串",
                "5. 注意：接口无法调用，只需要提供接口调用的计划即可。禁止重复调用接口。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_forward_summarizer(
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
                "4. 注意：接口无法调用，只需要提供接口调用的计划即可。禁止重复调用接口。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_forward_checker(
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
                "1. 判断最近的一个历史执行步骤是否合理完成了它在该步骤中设定的目标(即history_steps中的step_goal)。判断时遵循**抓大放小**原则",
                "2. 当前目标(current_goal)是下一步才做的事，不要用它来判断历史步骤的合理性。",
                "3. 你可以使用工具去检索额外的服务接口信息，来验证最近一个历史执行步骤的可行性。",
                "4. JSON格式：{'reasonable': ..., 'hint': ...}",
                "4.1 reasonable(bool): 最近的一步历史执行步骤是否合理地完成了它设定的目标(step_goal)，判断时遵循**抓大放小**原则"
                "4.2 hint(str): 如果合理，接下来如何完成当前目标(current_goal)的提示信息。如果不合理，重新执行上一步目标(step_goal)的提示信息。",
                "5. 注意：接口无法调用，只需要提供接口调用的计划即可。判断时遵循**抓大放小**原则。禁止重复调用接口。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_backward_planner(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        return Agent(
            name="Backward Plan Agent",
            model=model,
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="text-embedding-qwen3-embedding-8b",
                    enable_search_similar_interfaces=True,  # 允许搜索输出包含所需信息的接口
                )
            ],
            role="反向推导完成目标所需的前置步骤",
            instructions=[
                "1. 你是一个反向规划专家。你的任务是从结果倒推原因。",
                "2. Input中给了你'当前需解决的目标(Result)'。你需要找到一个服务接口，该接口的**输出(Output)**能满足这个Result。",
                "3. 找到接口后，分析运行该接口需要哪些**必要输入(Input)**。这些Input就构成了'Previous Goal'。",
                "4. 如果'用户原始问题'中已经包含了运行该接口所需的所有信息，则 previous_goal 设为空字符串（表示搜索结束）。",
                "5. JSON格式: {'interface_id': ..., 'plan': ..., 'previous_goal': ...}",
                "5.1 interface_id: 能产出当前Result的接口ID。",
                "5.2 plan: 描述如何使用该接口产出Result。",
                "5.3 previous_goal: 运行该接口所需的先决条件/数据。如果数据已在原始问题中，留空。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_backward_checker(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        return Agent(
            name="Backward Check Agent",
            model=model,
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="text-embedding-qwen3-embedding-8b",
                )
            ],
            role="检查反向推导的逻辑合理性",
            instructions=[
                "1. 检查最近生成的规划步骤：判断选用的接口是否真的能解决'Current Goal'。",
                "2. 检查'Previous Goal'是否准确描述了该接口的输入需求。",
                "3. 遵循'抓大放小'原则。只要逻辑链路通顺（Goal <- Output <- Interface <- Input <- Previous Goal）即可。",
                "4. JSON格式：{'reasonable': ..., 'hint': ...}",
                "4.1 reasonable(bool): 推导是否合理。",
                "4.2 hint(str): 如果合理，给出下一步继续倒推的提示；如果不合理，指出错误原因。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_backward_summarizer(
        uri: str,
        user: str,
        password: str,
        database: str,
        model: Model,
        embedding_base_url: str,
    ):
        # Summarizer 可以复用或微调，因为输入的数据结构已经被倒序整理好了，看起来和前向执行的结果一样
        return Agent(
            name="Backward Summarizer Agent",
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
            role="根据规划好的路径回答用户问题",
            instructions=[
                "1. 输入是一组已经规划好的、顺序的执行步骤（从Step 1到Step N）。",
                "2. 这些步骤是通过反向搜索生成的，现在请你按正向逻辑（Step 1 -> Final）整理成完整的执行方案。",
                "3. 详细描述每一步的计划和数据流转。",
            ],
            markdown=True,
        )
