# Local research pipeline

The four stages use Ollama or Gemini for source selection, writing and critique,
Tavily for web search, and Requests/BeautifulSoup for webpage reading.

## Setup and run

From the project directory in PowerShell:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 pipeline.py
~~~

For a new checkout, create the environment first with python -m venv .venv.
Create .env using .env.example and set TAVILY_API_KEY. Keep an existing
.env and its API keys when updating the project.

MODEL_PROVIDER defaults to ollama. Ollama must be running for that provider.
The local model defaults to llama3.2. To select another installed model, set:

~~~dotenv
OLLAMA_MODEL=llama3.1
~~~

To use Gemini instead, keep the existing API key and set:

~~~dotenv
MODEL_PROVIDER=gemini
GEMINI_MODEL=gemini-3.6-flash
~~~

Gemini accepts either GEMINI_API_KEY or GOOGLE_API_KEY. GEMINI_MAX_OUTPUT_TOKENS
defaults to 8192. Gemini requires network access and uses the account's API quota.
The pipeline records the actual model in its returned state; provider errors
propagate instead of silently switching to a different model.

Restart the pipeline after changing .env. Use ollama list to see installed
models, or ollama pull llama3.1 to download one if needed. LLaMA 3.1 8B can
produce more detailed analysis than LLaMA 3.2 3B, with higher memory use and
latency. A model change cannot repair missing source evidence.

OLLAMA_BASE_URL, OLLAMA_NUM_CTX (default 16384) and
OLLAMA_NUM_PREDICT (default 2048) are optional. SCRAPE_MAX_CHARS controls the
per-page excerpt limit (default 10000, minimum 1000). If increasing the number or
size of sources, also account for the model's input/output context budget.

For this English-language sample on a machine with limited GPU memory, an 8B
profile with shorter excerpts reduces CPU work:

~~~dotenv
OLLAMA_MODEL=llama3.1
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=1280
SCRAPE_MAX_CHARS=5000
~~~

Excerpt limits are recorded in the evidence. Choose a larger limit when the
needed findings occur deeper in a page.

## Evidence and failure handling

- Search/Reader agents return real tool results directly. The pipeline checks
  successful ToolMessage artifacts; it does not accept final agent prose as
  evidence of a search, scrape or package installation.
- When a model omits a tool call, the pipeline executes the tool itself.
- Reader receives all search URLs. Up to three successful webpages are
  collected; inaccessible URLs are logged and alternative sources are tried.
- Scraping imports Requests, checks HTTP status and rejects empty/JavaScript-only
  pages and unsupported binary documents such as PDFs.
- Empty main placeholders fall back to readable body text. Forms wrapping an
  article retain their content, and the longest main/article container is used.
- Each page carries its exact URL, redirect URL, retrieval timestamp, publication
  date when available, and actual cleaned text. Excerpts default to 10000
  characters per page, with truncation explicitly recorded.
- Writer and Critic receive the same webpage evidence. Search snippets and
  model-generated Reader summaries are not used as report evidence.
- Writer prompts request a subsection for each topic component and an evidence
  table with exact measurements, units, periods and source IDs. Unknown dates
  must not be converted into inferred quarters or publication dates.
- Critic prompts check topic coverage and cap scores below 7 for omitted central
  components or fabricated/contradicted numerical and date claims.
- Reports require source-ID citations and supplied URLs. Unknown citations or
  URLs cause a revision and, if still unresolved, stop the pipeline.
- Critic scores below 7/10 trigger one revision with the previous draft and
  feedback. A remaining low/unparseable score yields status="needs_review".
- If search or all scrapes fail, the pipeline stops before writing.

run_research_pipeline(topic, max_sources=3, max_revisions=1) returns the report,
critique, score, status, source text/metadata and scrape errors for inspection.
Citation checks verify source identifiers and URLs, not the semantic truth of
every claim. Review source support for important numerical or causal claims;
the same local model writes and critiques the report.

## Regression checks

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
~~~

These tests mock network/model responses and cover real-text extraction, HTTP
failures, ignored hallucinated tool repairs, alternative sources, evidence
handoff, citation rejection and the bounded revision loop.
