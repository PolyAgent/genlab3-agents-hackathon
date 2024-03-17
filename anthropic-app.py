import streamlit as st
import anthropic
from components.sidebar import sidebar
import openai
from VCPilot import VCPilot
import random


sidebar()

# Cloude-3-opus setup
client = anthropic.Client(api_key=st.secrets.anthropic.api_key)

import logging
import sys
import os
import time
import qdrant_client
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, Settings
import os
from qdrant_client import QdrantClient
import openai

# Set up environment variables
os.environ["QDRANT_URL"] = st.secrets.qdrant.url
os.environ["QDRANT_API_KEY"] = st.secrets.qdrant.api_key
os.environ["OPENAI_API_BASE"] = st.secrets.fireworks.base_url
os.environ["OPENAI_API_KEY"] = st.secrets.fireworks.api_key
os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic.api_key

vcpilot = VCPilot()

st.title("VC pilot")

if question := st.chat_input("How risky is this project?:"):
    st.chat_message("user").markdown(question)

    try:    
        # with st.spinner("Generating report..."):
        #     response = vcpilot.get_full_report(question)
        with st.spinner("Rephrasing problem statement..."):
            time.sleep(2)
            problem_statement = vcpilot.get_problem_statement(question)
        with st.spinner("Generating research tasks..."):
            tasks = vcpilot.get_research_tasks(question)
        with st.spinner("Initializing agent..."):
            time.sleep(2)
            agent_executor = vcpilot.get_agent_executor()
        with st.spinner("Agent performing research..."):
            summaries, citations = vcpilot.get_research(question, agent_executor, tasks)
        with st.spinner("Getting highlights from research..."):
            highlights = vcpilot.generate_highlights(question, citations, summaries)
        with st.spinner("Considering areas for followup..."):
            followups = vcpilot.get_followup_questions(highlights)
        with st.spinner("Wrapping up..."):
            conclusion = vcpilot.get_conclusion(question, highlights)
        tasks_str = "- " + "\n- ".join(tasks)
        # problem_statement_fix = problem_statement.replace("\"", "")
        final_report = f"""
## Problem Statement
{problem_statement_fix}

## Scope of Tasks
{tasks_str}

## Research
{highlights}

## Follow up Questions
{followups}

## Conclusion
{conclusion}
"""
        st.chat_message("assistant").markdown(final_report)
    except Exception as e:
        responses = [
            "It seems your idea isn't quite within the AI domain or is too vague for us to assess accurately.",
            "Your idea doesn't seem directly related to AI, or it might be too ambiguous for us to provide a clear evaluation.",
            "From what we can gather, your idea doesn't align with AI or it's too ambiguous to form a solid judgment.",
            "Your idea appears to be either unrelated to AI or too nebulous for us to discern its connection.",
            "It seems your idea is either not within the AI realm or too unclear for us to determine its relevance.",
            ]
        st.chat_message("assistant").markdown(random.choice(responses))
