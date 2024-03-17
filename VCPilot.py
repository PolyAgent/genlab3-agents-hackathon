import fire
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
import openai

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


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
            base_url = os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        response = client.chat.completions.create(
          model="accounts/fireworks/models/mixtral-8x7b-instruct",
          temperature=0,
          max_tokens=4096*4,
          messages=[{
            "role": "user",
            "content": prompt,
          }],
        )
        return response.choices[0].message.content

class VCPilot:
    def __init__(self):
        self.QDRANT_URL = os.environ["QDRANT_URL"]
        self.QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
        self.OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
        self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.COLLECTION_NAME = "vc-pilot-full"
        self.index, self.retriever = self.get_index_and_retriever()
        self.llm = FireworkLLM()


    def get_index_and_retriever(self):
        qdrant_client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        embed_model = FastEmbedEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        Settings.embed_model = embed_model

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=self.COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        retriever = index.as_retriever()
        return index, retriever
    
    def get_relevant_documents(self, question: str) -> List[str]:
        documents = self.retriever.retrieve(question)
        sorted_documents = sorted(documents, key=lambda node: node.score, reverse=True)
        relevant_documents = [node.text.replace("\n", " ") for node in sorted_documents]
        return relevant_documents

    def get_research_tasks(self, proposal: str) -> List[str]:
        research_template = PromptTemplate.from_template(
            template = """
        You are a research analyst for a venture capital investor that is evaluating AI technology startups.
        You have expertise in AI field and you understand how to evaluate the risk profile for new startup ideas.
        A startup is proposing the following idea:
        {proposal}

        Your task is to use the TRIZ methodology of problem solving to generate a list of the top 5 topics that would need to be researched.
        The list should be prioritized based on factors that can predict the success of the startup and should be simple, concise.
        This list will be used to search a database of 100,000 scientific papers on AI

        Here is an example of independent reserach topics that may be pursued where the topic is a statup's proposal for funding on a 1-bit quantization model:
        - Impact of 1-bit quantization on LLM performance and accuracy": Understanding the trade-off between computational efficiency and model accuracy is crucial to predict the success of the startup's proposal.
        - Scalability of 1-bit quantized LLMs in real-world applications": The startup's success depends on the applicability and scalability of their solution in various industries and use-cases.
        - Comparative analysis of 1-bit quantization with other model compression techniques": It's essential to evaluate how the proposed solution compares to alternative methods in terms of efficiency, accuracy, and ease of implementation.
        - Hardware and software requirements for implementing 1-bit quantized LLMs": Understanding the infrastructure needed to support the startup's solution will help assess the overall cost, accessibility, and feasibility of their proposal.
        - Case studies and success stories of 1-bit quantization in AI models": Learning from previous experiences and successes can provide valuable insights into potential challenges, opportunities, and the overall viability of the startup's idea.

        Respond in only `-` delimited list format with in depth independently operable research tasks.
        """
        )

        prompt = research_template.format(
            proposal=proposal,
        )

        response = self.llm.invoke(prompt)
        tasks = list(
                filter(
                    lambda y: y != "",
                    map(
                        lambda x: x.strip()\
                                    .replace("- ", "")\
                                    .replace("\"", ""),
                        response.split("\n")
                        )
                    )
                )
        return tasks

    def get_agent_executor(self) -> AgentExecutor:

        retriever_tool = create_retriever_tool(
            self.retriever,
            "document_retriever",
            "Search for information of latest computer science and language model research papers. For any questions about the latest research, use this tool.",
        )

        tools = [
            retriever_tool,
        ]

        agent_prompt = PromptTemplate(
            input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
            template="""
            Answer the following questions as best you can.
            You have access to the following tools that can reference the latest research on the given topic.
            Do not hallucinate.  If you do not have the relevant research to the question, exclusively say `NOT ENOUGH INFORMATION`:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}"""
        )

        agent = create_react_agent(
            self.llm, tools, agent_prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_execution_time=600
        )

        return agent_executor

    def get_research(self, proposal: str, agent_executor: AgentExecutor, tasks: List[str]) -> (List[str], List[str]):
        summaries = []
        citations = self.get_relevant_documents(proposal)
        for i, task in enumerate(tasks):
            response = agent_executor.invoke({"input": task})
            summary = response["output"]
            summaries.append(summary)
            # print(f"TASK {i+1}: {task}")
            # print(f"RESEARCH: {summary}")
            # # print(f"CONTENT {citations}")
            # print("#"*20)
        return summaries, citations

    def generate_highlights(self, proposal: str, citations: List[str], summaries: List[str]) -> str:
        highlights_prefix = f"""
    You are a research analyst working for a venture capital firm and you need to assess a risk profile of a deep tech startup that is working on the following proposal:
    {proposal}
    """

        highlights_chunks = ""
        for i, (citation, summary)  in enumerate(zip(citations, summaries)):
            highlights_chunks += f"""
    Citation {i+1}:
    {citation}

    Summary {i+1}:
    {summary}
    """

        
        highlights_suffix = """
    Using these citations and summaries only - please provide a final risk report analysis using this template:

### Identify Key Challenges
### Explore Potential Solutions
### Assess Innovation and Trend Alignment
### Solution Potential and Risk Evaluation

    ONLY OUTPUT CONTENT WITHIN THESE 4 SECTIONS
    Respond in only `-` delimited list format with 3-5 items in EACH SECTION.
    """
        highlights_prompt = "\n".join([highlights_prefix, highlights_chunks, highlights_suffix])
        response = self.llm.invoke(highlights_prompt)
        return response

    def get_followup_questions(self, highlights_response: str) -> List[str]:
        followup_questions_prompt = f"""
        You are a risk analyst working in a venture capital evaluating a tech startup. You have done the research on the technology and generated a risk report. Based on the risk report below, come up with 10 follow-up questions directed towards the startup founders. Be concise and critical or a kid would die. The questions cannot be answered by evaluating the risk report and should only address the important points. Limit the question to a sentence and less than 100 words.
        {highlights_response}
        """
        followup_response = self.llm.invoke(followup_questions_prompt)
        return followup_response

    def get_problem_statement(self, proposal: str) -> str:
        problem_statement_prompt = f"""
        Given a press release from a startup, create a statement describing problem and proposed solution that the startup is trying to address in detail. Be concise and targeted or kids would die. Only return the statement, and keep the statement one sentence. Summarize the information and do not include specific numbers or data.
        {proposal}
        """
        problem_statement = self.llm.invoke(problem_statement_prompt)
        # print(problem_statement)
        return problem_statement

    def get_conclusion(self, proposal: str, highlights_response: str) -> str:
        conclusion_prompt = f"""
You are a research analyst working on a due diligence report for a venture capital firm and need to assess a technology risk profile of a deep tech AI startup that is working on
{proposal}

You've read all the research papers relevant to this topic, produced a preliminary report, formulated relevant questions and want to create a due diligence conclusion summary based on that. Please write a conclusion section for the report, focus on investment risks of our firm, a venture capital firm investing into early stage startups, keep it under 200 words and focus on technology risk. Be really critical analyst, like someone's job depends on this. Be dry and to the point, less verbose and more factual. And then include one-liner to describe the risk profile like you are describing it to a friend, super simple and with a bit of a humor.
{highlights_response}

Respond in the following format:
Conclusion: <serious conclusion summary>
Final Thoughts: <goofy analogy to a friend>
Rating: <1.0 - 10.0>
Emoji: <relevant emoji>
        """

        conclusion = self.llm.invoke(conclusion_prompt)
        return conclusion

    def get_full_report(self, proposal: str) -> str:
        problem_statement = self.get_problem_statement(proposal)
        tasks = self.get_research_tasks(proposal)
        citations = self.get_relevant_documents(proposal)
        agent_executor = self.get_agent_executor()
        summaries, citations = self.get_research(proposal, agent_executor, tasks)
        highlights = self.generate_highlights(proposal, citations, summaries)
        followup_response = self.get_followup_questions(highlights)
        conclusion = self.get_conclusion(proposal, highlights)
        tasks_str = "- " + "\n- ".join(tasks)
        final_report = f"""
## Problem Statement
{problem_statement}

## Scope of Tasks
{tasks_str}

## Research
{highlights}

## Follow up Questions
{followup_response}

## Conclusion
{conclusion}
        """
        return final_report

    def run(self, proposal: str):
        final_report = self.get_full_report(proposal)
        print(final_report)

if __name__ == "__main__":
    fire.Fire(VCPilot)
