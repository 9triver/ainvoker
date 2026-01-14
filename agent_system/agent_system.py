import sys, os

sys.path.append(os.getcwd())

from typing import List
from pydantic import BaseModel, Field
from agno.models.base import Model
from agno.agent import Agent, RunOutput
from agno.utils.pprint import pprint_run_response

from tools.service import ServiceTools
from .world_state import WorldState
from .actions import InterfaceAction


class SearchResult(BaseModel):
    interface_ids: List[str] = Field(
        ..., description="List of interface IDs that can help solve the user's problem"
    )
    requied_entities: List[str] = Field(
        ...,
        description="List of descriptions for business entities that need to be supplemented or identified next",
    )


class AgentSystem:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        search_model: Model,
        summarize_model: Model,
        embedding_base_url: str,
    ) -> None:
        self.interface_action = InterfaceAction(
            uri=uri, user=user, password=password, database=database
        )
        self.searcher = AgentSystem.init_searcher(
            model=search_model,
            uri=uri,
            user=user,
            password=password,
            database=database,
            embedding_base_url=embedding_base_url,
        )
        self.summarizer = AgentSystem.init_summarizer(
            model=summarize_model,
            uri=uri,
            user=user,
            password=password,
            database=database,
            embedding_base_url=embedding_base_url,
        )
        return

    def response(self, question: str, max_step: int = 10) -> str:
        world_state = WorldState(
            origin_question=question,
        )

        step_num = 0
        interface_id_history = set()
        while step_num < max_step:
            response: RunOutput = self.searcher.run(
                input=str(world_state),
                debug_mode=True,
            )
            pprint_run_response(response, markdown=True)
            search_result: SearchResult = response.content
            interface_ids = search_result.interface_ids
            requied_entities = search_result.requied_entities

            world_state = self.interface_action.update_by_interface_ids(
                state=WorldState(
                    origin_question=question, required_entities=requied_entities
                ),
                interface_ids=interface_ids,
            )

            if (
                interface_id_history
                and interface_ids
                and all(
                    interface_id in interface_id_history
                    for interface_id in interface_ids
                )
            ):
                break
            interface_id_history.update(interface_ids)

            step_num += 1

        response: RunOutput = self.summarizer.run(
            input=str(world_state),
            debug_mode=True,
        )
        pprint_run_response(response, markdown=True)
        return response.content

    @staticmethod
    def init_searcher(
        model: Model,
        uri: str,
        user: str,
        password: str,
        database: str,
        embedding_base_url: str,
    ) -> Agent:
        return Agent(
            name="Search Agent",
            model=model,
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="nvidia-llama-embed-nemotron-8b",
                    enable_search_similar_output_entities=True,
                    enable_search_similar_cim_classes=True,
                )
            ],
            role="分析用户问题和候选接口，继续搜索所需的业务实体和接口",
            instructions=[
                "1. 我提供了用户原始问题和候选接口信息。",
                "2. search_similar_output_entities: 可以写一个文本语句来描述需要的业务实体，并搜索到目标业务实体及其接口信息",
                "3. search_similar_cim_classes: 同样可以搜索到CIM类信息，它可以给你某些实体明确的定义",
                "4. 分析我提供的候选接口信息，和目标业务实体的接口信息，保留所有有助于解决用户问题的接口。",
            ],
            output_schema=SearchResult,
            use_json_mode=True,
            markdown=True,
        )

    @staticmethod
    def init_summarizer(
        model: Model,
        uri: str,
        user: str,
        password: str,
        database: str,
        embedding_base_url: str,
    ) -> Agent:
        return Agent(
            name="Summarize Agent",
            model=model,
            role="分析用户问题，候选接口和需要的业务实体，制定解决方案",
            tools=[
                ServiceTools(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                    embedding_base_url=embedding_base_url,
                    embedding_model="nvidia-llama-embed-nemotron-8b",
                    enable_search_similar_cim_classes=True,
                )
            ],
            instructions=[
                "1. 我提供了用户原始问题，候选接口信息，和可能需要的业务实体描述。",
                "2. 给用户一个详细的解决方案，包括调用哪些接口，介绍接口信息，要求用户补充具体的业务实体信息。",
                "3. 写解决方案时，给用户填写接口调用示例，填入输入和输出信息，让用户明白应该如何调用接口。",
                "4. 当缺少接口时，你来补充一个适当的接口。为了使补充接口更加严谨，可以使用工具去收集相关CIM类信息作为参考。"
            ],
            markdown=True,
        )
