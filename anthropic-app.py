import streamlit as st
import anthropic
from components.sidebar import sidebar
import requests
from bs4 import BeautifulSoup
import re
# The URL of the article you want to scrape
def scrape_techcrunch_content(article_url):
    try:
        # Fetch the content of the article
        response = requests.get(article_url)
        response.raise_for_status()  # Raises an error for bad responses
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the main content of the article
        # The class name will vary by site, inspect the article to find the right one
        article_body = soup.find('div', class_='article-content')  # Adjust class as needed
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = '\n'.join([paragraph.get_text() for paragraph in paragraphs])
            return article_text
        else:
            return "Article content could not be found."
    except requests.RequestException as e:
        return f"Request failed: {e}"
def fecthCompanyName(url):
    match = re.search(r'https://www\.(.*?)\.', url)
    if match:
        # The extracted text is in the first group of the match
        name = match.group(1)
    else:
        print("No match found.")
def get_startup_description(url):
    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP request errors
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        companyName = str(fecthCompanyName(url))
        # Attempt to find sections that likely contain the startup description
        # This includes common identifiers like "about", "mission", "what-we-do"
        # some keywords are hardcoded here to make it work for the hack
        keywords = ['about', 'mission', 'what we do', 'story', 'protein','cradle']
        keywords.append(companyName)
        description_text = []
        for keyword in keywords:
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'div'], string=lambda text: text and keyword.lower() in text.lower()):
                # Find the parent or next siblings of the tag that might contain the description
                parent = tag.find_parent()
                if parent and parent not in description_text:
                    description_text.append(parent.get_text(separator=".", strip=True))
                else:
                    for sibling in tag.next_siblings:
                        if sibling not in description_text:
                            description_text.append(sibling.get_text(separator=".", strip=True))
                break  # Break after finding the first relevant section to avoid duplicates
        # Combine and return the collected texts
        return ' '.join(description_text) if description_text else "Could not find a startup description."
    except requests.RequestException as e:
        return f"Failed to fetch the webpage: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

import openai
from VCPilot import VCPilot


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

vcpilot = VCPilot()

st.title("VC pilot Claude-3-opus")

example_articles = [
    "https://techcrunch.com/2024/03/12/axion-rays-ai-attempts-to-detect-product-flaws-to-prevent-recalls/",
    "https://techcrunch.com/2023/11/09/ghost-now-openai-backed-claims-llms-will-overcome-self-driving-setbacks-but-experts-are-skeptical/",
    "https://techcrunch.com/2022/02/02/scale-ai-gets-into-the-synthetic-data-game/"
]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Axion Ray"):
        st.session_state['selected_article_url'] = example_articles[0]
with col2:
    if st.button("Ghost Autonomy"):
        st.session_state['selected_article_url'] = example_articles[1]
with col3:
    if st.button("Scale AI"):
        st.session_state['selected_article_url'] = example_articles[2]

techcrunch_article_url = st.text_input("Enter a TechCrunch article URL:", value=st.session_state.get('selected_article_url', ''))
if techcrunch_article_url:
    scraped_content = scrape_techcrunch_content(techcrunch_article_url)
    if scraped_content:
        st.text_area("Scraped Article Content:", scraped_content, height=300)
        # Proceed to use the scraped content as input for your analysis
        with st.spinner("Analyzing article content..."):
            # Simulate analysis of the article content
            problem_statement = vcpilot.get_problem_statement(scraped_content)  # Assuming this function can take the article content
            # Additional processing based on the article content
            # This part replaces the manual question input and directly uses the article content

            # You can replace or adjust the logic below according to how you process the content
            tasks = vcpilot.get_research_tasks(scraped_content)
            with st.spinner("Initializing agent..."):
                time.sleep(2)  # Simulated delay for demonstration
                agent_executor = vcpilot.get_agent_executor()
            summaries, citations = vcpilot.get_research(scraped_content, agent_executor, tasks)
            highlights = vcpilot.generate_highlights(scraped_content, citations, summaries)
            followups = vcpilot.get_followup_questions(highlights)
            conclusion = vcpilot.get_conclusion(scraped_content, highlights)
            
            # Constructing the final report
            tasks_str = "- " + "\n- ".join(tasks)
            final_report = f"""
## Problem Statement
{problem_statement}

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

