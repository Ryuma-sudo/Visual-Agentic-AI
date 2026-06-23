from __future__ import annotations

from langchain.agents import create_agent

from src.llm import build_llm
from src.tools.research_tools import build_research_tools


def build_research_agent():
    return create_agent(
        model=build_llm(),
        tools=build_research_tools(),
        system_prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks.\n"
            "- Use Arxiv for scientific papers and Wikipedia for general knowledge.\n"
            "- After you are done, respond to the supervisor directly.\n"
            "- Respond ONLY with the results of your work, do not include extra chatter."
        ),
        name="research_agent",
    )

