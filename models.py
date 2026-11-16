import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


load_dotenv(Path(__file__).resolve().parent / ".env")


def build_llm(provider: str | None = None) -> BaseChatModel:
    """Select a configured model without silently falling back to another provider."""
    provider = (provider or os.getenv("MODEL_PROVIDER", "ollama")).strip().lower()
    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "2048")),
            client_kwargs={"timeout": 300},
        )

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini requires GOOGLE_API_KEY or GEMINI_API_KEY in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 3.x uses its default sampling settings.
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            google_api_key=api_key,
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192")),
            timeout=120,
            max_retries=1,
        )

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq requires GROQ_API_KEY in .env.")
        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        options = {
            "model": model,
            "groq_api_key": api_key,
            "temperature": 0,
            "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "2048")),
            "timeout": 120,
            "max_retries": 2,
        }
        if model.startswith("openai/gpt-oss"):
            options["reasoning_effort"] = "low"
        return ChatGroq(**options)

    raise ValueError(f"Unsupported MODEL_PROVIDER '{provider}'. Choose ollama, gemini or groq.")
