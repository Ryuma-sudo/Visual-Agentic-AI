from __future__ import annotations

import _bootstrap  # noqa: F401

from src.agents.research_agent import build_research_agent
from src.utils.printing import pretty_print_messages


if __name__ == "__main__":
    agent = build_research_agent()
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Rotary Positional Encoding"}]},
        config={"recursion_limit": 50},
    ):
        pretty_print_messages(chunk, last_message=True)
