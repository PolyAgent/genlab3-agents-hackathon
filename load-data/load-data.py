
import logging
import sys
import os

import qdrant_client
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Document, Settings
import os
from qdrant_client import QdrantClient
import openai

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from tqdm.auto import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Lock

COLLECTION_NAME = "vc-pilot-full"

def get_indexer_and_retriever():
  # Set up Qdrant client for vector store
    qdrant_client = QdrantClient(
        url='https://56ab7b97-f618-4723-9b13-93e0b140c31b.us-east4-0.gcp.cloud.qdrant.io:6333',
        api_key="",
    )

    # Embedding model for vector insertion
    os.environ["OPENAI_API_BASE"]="https://api.fireworks.ai/inference/v1"
    os.environ["OPENAI_API_KEY"] =""
    embed_model = FastEmbedEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
    Settings.embed_model = embed_model

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    retriever = index.as_retriever()
    return index, retriever

index, retriever = get_indexer_and_retriever()

class MongoDBQuery:
    def __init__(self, db_name, collection_name, uri="mongodb://localhost:27017/"):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(self.uri)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
        except ConnectionFailure:
            print("Failed to connect to MongoDB")
            raise

    def query(self, query_filter=None):
        if query_filter is None:
            query_filter = {}
        try:
            results = self.collection.find(query_filter)
            return list(results)
        except Exception as e:
            print(f"An error occurred during the query: {e}")
            raise


class FireworkLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "Firework"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = openai.OpenAI(
            base_url = "https://api.fireworks.ai/inference/v1",
            api_key="",
        )
        response = client.chat.completions.create(
          model="accounts/fireworks/models/mixtral-8x7b-instruct",
          temperature=0,
          max_tokens=16000,
          messages=[{
            "role": "user",
            "content": prompt,
          }],
        )
        return response.choices[0].message.content


def get_mongo_content():
    mongo = MongoDBQuery(
        db_name="arxiv",
        collection_name="papers_for_review",
        uri="mongodb+srv://genlab-hackathon:qSzbc3NWGgWie1aP@age-house.dypq7r5.mongodb.net",
    )
    mongo_content = mongo.query(query_filter={"abstract": {"$exists": True}})
    print("Doc count", len(mongo_content))
    return mongo_content


llm = FireworkLLM()
mongo_content = get_mongo_content()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

def process_paper(lock, progress_bar, paper, use_abstracts=True, use_papers=False):
    if use_abstracts:
        title = paper["title"]
        text_content = paper["abstract"]
        doc = Document(text=f"{title}:\n{text_content}")
        index.insert(doc)
    if use_papers:
        pdf_loader = PyPDFLoader(paper["pdf_url"])
        documents = pdf_loader.load_and_split(text_splitter=text_splitter)
        index.insert(documents)
    with lock:
        progress_bar.update(1)

lock = Lock()

max_workers = os.cpu_count()
progress_bar = tqdm(total=len(mongo_content))
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    use_abstracts = True
    use_papers = False
    futures = [executor.submit(process_paper, lock, progress_bar, paper, use_abstracts, use_papers) for paper in mongo_content]

    for future in as_completed(futures):
        pass

progress_bar.close()
