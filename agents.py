import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url, get_secret
from dotenv import load_dotenv

load_dotenv()

routix_api_key = get_secret("ROUTIX_API_KEY") or "missing_key"

llm = ChatOpenAI(
    model="deepseek-v4-flash-0731", # Replace with whatever model Routix provides
    api_key=routix_api_key,
    base_url="https://routixai.online/v1", 
    temperature=0,
    max_retries=5
)    


#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search],
        system_prompt=(
            "You are an exhaustive research agent. Run at least 4 different searches: "
            "1) latest news, 2) statistics and numbers, 3) criticism and risks, 4) expert or official reports. "
            "Gather 6-10 distinct, high-quality sources. Never use Wikipedia. "
            "Return each source as: [n] TITLE | URL | DATE | key facts with numbers."
        )
    )

#2nd agent 
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        system_prompt=(
            "You are an expert data extractor. Scrape 3-4 of the best URLs (different websites). "
            "For each URL return: SOURCE: url, KEY FACTS: exact numbers, metrics, costs, dates, examples, "
            "RISKS: any criticism or limits. Never drop the numbers."
        )
    )

#3rd agent - fact checker 
def build_factcheck_agent():
    return create_agent(
        model = llm,
        tools = [web_search],
        system_prompt=(
            "You are a fact-checker. Check every number and claim in the report "
            "against the research (search if needed). Output:\n"
            "VERIFIED: claim [url]\n"
            "WRONG: claim -> correct value [url]\n"
            "UNSUPPORTED: claim"
        )
    )


#writer chain 
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports. "
               "Use ONLY facts from the research. Never invent numbers."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 4 well-explained points)
- Risks & Limitations (real counterpoints, failures, open problems)
- Conclusion (new implications + 3 actionable recommendations, do NOT repeat the findings)
- Sources (numbered full URLs, minimum 4, no Wikipedia)

Rules:
- Cite every claim inline like [1], [2] matching the Sources list.
- Include at least 6 concrete numbers (stats, costs, dates, benchmarks).
- If the research includes CRITIC FEEDBACK or FACT CHECK, fix every point mentioned.

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 
critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly on: 
sourcing, data and numbers, critical analysis, structure, and conclusion quality.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()