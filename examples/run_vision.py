from __future__ import annotations

import _bootstrap  # noqa: F401

from src.agents.vision_agent import build_vision_agent
from src.utils.printing import pretty_print_messages


if __name__ == "__main__":
    agent = build_vision_agent()
    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Describe this image: "
                        "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"
                    ),
                }
            ]
        },
        config={"recursion_limit": 50},
    ):
        pretty_print_messages(chunk, last_message=True)
