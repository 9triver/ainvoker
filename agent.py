import sys, os
sys.path.append(os.getcwd())
from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.models.openai import OpenAILike
from agno.db.sqlite import SqliteDb
from service_list.tools.service import ServiceTools

sys.path.append(os.getcwd())

agent = Agent(
    name="Agno Agent",
    model=OpenAILike(
        api_key=os.getenv("MODELSCOPE_API_KEY"),
        base_url="https://api-inference.modelscope.cn/v1",
        id="ZhipuAI/GLM-4.6",
    ),
    tools=[
        ServiceTools(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="12345678",
            database="service-list",
            embedding_model="../model/Qwen/Qwen3-Embedding-4B",
            enable_search_interfaces=True,
        )
    ],
    db=SqliteDb(db_file="./tmp/data.db"),
    add_history_to_context=True,
    markdown=True,
    instructions="找到相关服务接口，用中文回答用户问题。用户不会提供任何数据，只需要告知用户依次调用哪些服务接口，给出具体的解决方案即可。",
)

if __name__ == "__main__":
    agent.cli_app(stream=True)
