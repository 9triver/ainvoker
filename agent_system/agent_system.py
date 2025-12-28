import sys, os

sys.path.append(os.getcwd())

from agno.models.base import Model
from agno.agent import Agent, RunOutput
from agno.utils.pprint import pprint_run_response
from json_repair import loads

from tools.service import ServiceTools
from .world_state import WorldState
from .actions import InterfaceAction


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
        self.interface_action = InterfaceAction(
            uri=uri, user=user, password=password, database=database
        )
        self.searcher = AgentSystem.init_searcher(
            model=model,
            uri=uri,
            user=user,
            password=password,
            database=database,
            embedding_base_url=embedding_base_url,
        )
        self.summarizer = AgentSystem.init_summarizer(
            model=model,
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
            response_obj = loads(response.content)
            interface_ids = response_obj["interface_ids"]
            requied_entities = response_obj["requied_entities"]

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
                    embedding_model="tencent-kalm-embedding-gemma3-12b-2511",
                    enable_search_similar_output_entities=True,
                )
            ],
            role="分析用户问题和候选接口，继续搜索所需的业务实体和接口",
            instructions=[
                "1. 我提供了用户原始问题和候选接口信息。根据下面的思路去寻找接下来你需要的业务实体和接口信息:",
                "1.1 生成需要的业务实体的描述，使用工具进行搜索与描述相关的业务实体。推荐多尝试不同描述来搜索。",
                "1.2 如果多次搜索尝试后仍未找到有助于直接或间接解决用户问题的业务实体，填写需要业务实体(requied_entities)列表。",
                "2. 根据我提供的候选接口信息，和目标业务实体的接口信息，保留所有有助于解决用户问题的接口。",
                "3. JSON格式: {'interface_ids': [...], 'requied_entities': [...]}",
                "3.1 interface_ids(list[str]): 所有有助于解决用户问题的接口ID列表，在更合适的接口出现之前，可以接受相对不完美的有用接口。",
                "3.2 requied_entities(list[str]): 需要我补充的、以及接下来需要寻找的业务实体的描述列表。",
            ],
            markdown=True,
        )

    @staticmethod
    def init_summarizer(
        model: Model,
    ) -> Agent:
        return Agent(
            name="Summarize Agent",
            model=model,
            role="分析用户问题，候选接口和需要的业务实体，制定解决方案",
            instructions=[
                "1. 我提供了用户原始问题，候选接口信息，和可能需要的业务实体描述。",
                "2. 给用户一个详细的解决方案，包括调用哪些接口，介绍接口信息，要求用户补充具体的业务实体信息。",
                "3. 写解决方案时，给用户填写接口调用示例，填入输入和输出信息，让用户明白应该如何调用接口。",
            ],
            markdown=True,
        )
