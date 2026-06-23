from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


def build_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Create code/.env from .env.example and add your key."
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )
