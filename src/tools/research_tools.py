from __future__ import annotations

import wikipedia as wikipedia_pkg
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

from src.config import settings


def build_research_tools():
    wikipedia_pkg.set_user_agent(settings.wikipedia_user_agent)

    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000),
        description="Search for scientific papers on a topic using Arxiv.",
    )
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000),
        description="Search for general information on a topic using Wikipedia.",
    )
    return [arxiv, wikipedia]
