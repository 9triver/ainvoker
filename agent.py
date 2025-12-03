import sys, os

sys.path.append(os.getcwd())

from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.models.openai import OpenAILike
from agno.db.sqlite import SqliteDb
from tools.service import ServiceTools


if __name__ == "__main__":
    base_url = os.getenv("LM_STUDIO_BASE_URL")
    agent = Agent(
        name="Agno Agent",
        # model=OpenAILike(
        #     api_key=os.getenv("MODELSCOPE_API_KEY"),
        #     base_url="https://api-inference.modelscope.cn/v1",
        #     id="ZhipuAI/GLM-4.6",
        # ),
        model=LMStudio(
            id="glm-4.6",
            base_url=base_url,
        ),
        tools=[
            ServiceTools(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="12345678",
                database="service-list-2025-11-26T07-05-38",
                base_url=base_url,
                embedding_model="text-embedding-qwen3-embedding-8b",
                enable_search_interfaces=True,
            )
        ],
        db=SqliteDb(db_file="./tmp/data.db"),
        add_history_to_context=True,
        markdown=True,
        # instructions="查找库中的CIMClass,给出相关类的信息和属性。",
        instructions="找到相关服务接口，用中文回答用户问题。用户不会提供任何数据，只需要告知用户依次调用哪些服务接口，给出具体的解决方案即可。如果有不存在但需要的接口，找到可能需要的CIM模型类，模拟出假接口，如果没有对应类，请给出需要补充的实体。",
    )
    agent.cli_app(stream=True)
