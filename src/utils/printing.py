from __future__ import annotations

from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent: bool = False) -> None:
    pretty_message = message.pretty_repr(html=False)
    if indent:
        pretty_message = "\n".join("\t" + line for line in pretty_message.splitlines())
    print(pretty_message)


def pretty_print_messages(update, last_message: bool = False) -> None:
    is_subgraph = False
    if isinstance(update, tuple):
        namespace, update = update
        if len(namespace) == 0:
            return
        graph_id = namespace[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        label = f"Update from node {node_name}:"
        if is_subgraph:
            label = "\t" + label
        print(label + "\n")

        messages = convert_to_messages(node_update.get("messages", []))
        if last_message:
            messages = messages[-1:]
        for message in messages:
            pretty_print_message(message, indent=is_subgraph)
        print()
