from __future__ import annotations


def get_final_messages(last_chunk):
    if last_chunk is None:
        return []
    if isinstance(last_chunk, tuple):
        _, last_chunk = last_chunk
    if isinstance(last_chunk, dict) and "supervisor" in last_chunk:
        return last_chunk["supervisor"].get("messages", [])
    if isinstance(last_chunk, dict):
        for node_update in last_chunk.values():
            if isinstance(node_update, dict) and "messages" in node_update:
                return node_update["messages"]
    return []


def get_final_text(last_chunk) -> str:
    messages = get_final_messages(last_chunk)
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    return content if isinstance(content, str) else str(content)
