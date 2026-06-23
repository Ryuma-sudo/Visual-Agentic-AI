from __future__ import annotations

import sys

from src.supervisor import build_supervisor
from src.utils.messages import get_final_text
from src.utils.printing import pretty_print_messages


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python main.py "your question"')
        return 1

    user_input = " ".join(sys.argv[1:])
    supervisor = build_supervisor()
    config = {"recursion_limit": 50}

    final_chunk = None
    for chunk in supervisor.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    ):
        final_chunk = chunk
        pretty_print_messages(chunk, last_message=True)

    final_text = get_final_text(final_chunk)
    if final_text:
        print("\nFinal answer:\n")
        print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
