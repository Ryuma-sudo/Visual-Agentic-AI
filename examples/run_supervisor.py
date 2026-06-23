from __future__ import annotations

import _bootstrap  # noqa: F401

from src.supervisor import build_supervisor
from src.utils.printing import pretty_print_messages


if __name__ == "__main__":
    supervisor = build_supervisor()
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the latest research on positional embeddings?",
                }
            ]
        },
        config={"recursion_limit": 50},
    ):
        pretty_print_messages(chunk, last_message=True)
