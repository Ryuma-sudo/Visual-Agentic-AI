from __future__ import annotations

from langgraph_supervisor import create_supervisor

from src.agents.research_agent import build_research_agent
from src.agents.vision_agent import build_vision_agent
from src.llm import build_llm


def build_supervisor():
    research_agent = build_research_agent()
    vision_agent = build_vision_agent()

    return create_supervisor(
        model=build_llm(),
        agents=[research_agent, vision_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- research_agent: Use ONLY for research tasks, scientific papers, "
            "Wikipedia-style facts, and concept explanations.\n"
            "- vision_agent: Use ONLY for image tasks, including describing images, "
            "detecting objects, and counting objects.\n\n"
            "RULES:\n"
            "- Assign tasks to only one agent at a time.\n"
            "- Do not call multiple agents in parallel.\n"
            "- Do not perform specialist work yourself; delegate it.\n"
            "- For mixed image + research questions, delegate to vision_agent first, "
            "then research_agent if more factual background is needed.\n"
            "- Keep handoff messages concise."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
