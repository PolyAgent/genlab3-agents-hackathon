import streamlit as st
import anthropic
from components.sidebar import sidebar
import openai
from VCPilot import VCPilot


sidebar()

# Cloude-3-opus setup
client = anthropic.Client(api_key=st.secrets.anthropic.api_key)

import logging
import sys
import os

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

vcpilot = VCPilot()

st.title("VC pilot Claude-3-opus")

if question := st.chat_input("How risky is this project?:"):
    st.chat_message("user").markdown(question)
    
    with st.spinner("Generating report..."):
        response = vcpilot.get_full_report(question)
    with st.chat_message("assistant"):
        st.markdown(response)