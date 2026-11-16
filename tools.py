from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    val = None
    # Check st.secrets first so dashboard updates take priority
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            val = st.secrets.get(key) or st.secrets.get(key.lower()) or st.secrets.get(key.upper())
    except Exception:
        pass

    # Fall back to environment variable
    if not val:
        val = os.getenv(key) or os.getenv(key.lower()) or os.getenv(key.upper())

    if val:
        # Strip whitespace and accidental surrounding quotes
        return str(val).strip().strip("'\"")
    return default

def get_tavily_client():
    key = get_secret("TAVILY_API_KEY")
    return TavilyClient(api_key=key or "missing_key")

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    try:
        tavily = get_tavily_client()
        results = tavily.search(query=query, max_results=5)
        out = []
        for r in results.get('results', []):
            out.append(
                f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nSnippet: {r.get('content', '')[:300]}\n"
            )
        return "\n----\n".join(out)
    except Exception as e:
        return f"Web search error: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
