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

techcrunch_article_url = st.text_input("Enter a TechCrunch article URL:", "")
if techcrunch_article_url:
    scraped_content = scrape_techcrunch_content(techcrunch_article_url)
    if scraped_content:
        st.text_area("Scraped Article Content:", scraped_content, height=300)
    else:
        st.error("Could not scrape content from the URL. Please check the URL and try again.")

if question := st.chat_input("How risky is this project?:"):
    st.chat_message("user").markdown(question)
    

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

