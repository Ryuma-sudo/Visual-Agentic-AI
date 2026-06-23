from __future__ import annotations

from langchain.agents import create_agent

from src.llm import build_llm
from src.tools.image_describer import ImageDescriberTool
from src.tools.object_detector import detect_and_count_objects


def build_vision_agent():
    return create_agent(
        model=build_llm(),
        tools=[ImageDescriberTool(), detect_and_count_objects],
        system_prompt=(
            "You are a vision agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with visual tasks such as describing images, detecting objects, "
            "and counting objects.\n"
            "- Use only the provided tools to analyze visual inputs.\n"
            "- After you are done, respond to the supervisor directly.\n"
            "- Respond ONLY with the results of your work, do not include extra chatter."
        ),
        name="vision_agent",
    )

