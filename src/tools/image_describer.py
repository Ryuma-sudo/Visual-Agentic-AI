from __future__ import annotations

from typing import Optional

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.llm import build_llm
from src.tools.image_utils import encode_image, extract_image_path_or_url


class ImageDescriberInput(BaseModel):
    text: str = Field(description="Path or URL to an image.")


class ImageDescriberTool(BaseTool):
    name: str = "image_describer"
    description: str = "Describe the visible content of an image in detail."
    args_schema: type[BaseModel] = ImageDescriberInput

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        image_path_or_url = extract_image_path_or_url(text)
        encoded = encode_image(image_path_or_url, get_mime_type=True)
        if not encoded or encoded[0] is None or encoded[1] is None:
            return (
                f"Cannot encode image from '{image_path_or_url}'. "
                "Check that the file or URL exists and is an image."
            )

        image_b64, image_mime_type = encoded
        llm = build_llm()
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an expert image describer. Provide a detailed, accurate, "
                        "objective description of visible content. Focus on objects, "
                        "positions, colors, composition, actions, and readable text. "
                        "Avoid unsupported speculation."
                    )
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": f"data:{image_mime_type};base64,{image_b64}",
                        },
                    ]
                ),
            ]
        )
        return str(response.content)

    async def _arun(
        self,
        text: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(text, run_manager=run_manager)
