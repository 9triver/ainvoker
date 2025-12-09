import sys, os

sys.path.append(os.getcwd())

from agno.models.openai import OpenAILike
from agent_system import AgentSystem
from prompt_toolkit import prompt, print_formatted_text


if __name__ == "__main__":
    base_url = os.getenv("EMBED_BASE_URL")
    model = OpenAILike(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        id="deepseek-chat",
    )

    agent_system = AgentSystem(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="12345678",
        database="service-cim",
        model=model,
        embedding_base_url=base_url,
    )
    while True:
        question = prompt("User: ")
        anwser = agent_system.response(question=question)
        print_formatted_text(anwser)
