import streamlit as st
import anthropic
from components.sidebar import sidebar
import openai

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

COLLECTION_NAME = "vc-pilot-full"

# Set up Qdrant client for vector store
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# Embedding model for vector insertion
from llama_index.embeddings.openai import OpenAIEmbedding

fw_embed_model = OpenAIEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    api_base=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"])
Settings.embed_model = fw_embed_model


vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=fw_embed_model
)
retriever = index.as_retriever()

st.title("VC pilot Claude-3-opus")

if question := st.chat_input("How risky is this project?:"):
    st.chat_message("user").markdown(question)
    
    question_context = retriever.retrieve(question)[0].text
    st.write(f"qdrant context: {question_context}")

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"""{question}. You can use next documents: {question_context}"""}
        ]
    )
    with st.chat_message("assistant"):
        st.markdown(response.content[0].text)