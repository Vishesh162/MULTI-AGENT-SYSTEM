import os
import unittest
from unittest.mock import Mock, patch

import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from tools import MAX_SCRAPED_CHARS, ScrapeError, scrape_url, web_search


URL = "https://source.example/report"
TEXT = (
    "On 12 March 2025 the evaluation processed 1,200 requests at a median latency "
    "of 42 ms on one CPU, using a 30-second client timeout. These measurements "
    "apply to the stated test environment only."
)


def response(html: str, content_type: str = "text/html") -> Mock:
    result = Mock()
    result.content = html.encode("utf-8")
    result.text = html
    result.url = URL
    result.headers = {"Content-Type": content_type}
    return result


def invoke(tool, args):
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "test-call"})


class ScraperTests(unittest.TestCase):
    @patch("tools.requests.get")
    def test_page_without_cleanup_tags_returns_actual_text(self, get):
        get.return_value = response(f"<html><body><p>{TEXT}</p></body></html>")
        result = invoke(scrape_url, {"url": URL})
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.artifact["text"], TEXT)
        get.return_value.raise_for_status.assert_called_once()

    @patch("tools.requests.get")
    def test_all_boilerplate_is_removed_before_returning(self, get):
        get.return_value = response(
            '<html><head><title>Measured latency</title>'
            '<meta property="article:published_time" content="2025-03-12"></head>'
            f'<body><script>FIRST_REMOVED</script><main><p>{TEXT}</p>'
            '<nav>NAVIGATION_NOISE</nav><footer>FOOTER_NOISE</footer>'
            '<style>STYLE_NOISE</style></main></body></html>'
        )
        result = invoke(scrape_url, {"url": URL})
        self.assertEqual(result.artifact["text"], TEXT)
        self.assertEqual(result.artifact["published_at"], "2025-03-12")
        self.assertEqual(result.artifact["title"], "Measured latency")

    @patch("tools.requests.get")
    def test_empty_main_placeholder_does_not_hide_body_article(self, get):
        get.return_value = response(f"<body><main></main><div><p>{TEXT}</p></div></body>")
        result = invoke(scrape_url, {"url": URL})
        self.assertEqual(result.artifact["text"], TEXT)

    @patch("tools.requests.get")
    def test_form_wrapping_article_keeps_its_research_text(self, get):
        get.return_value = response(f'<body><form method="post"><p>{TEXT}</p></form></body>')
        result = invoke(scrape_url, {"url": URL})
        self.assertEqual(result.artifact["text"], TEXT)

    @patch("tools.requests.get")
    def test_http_error_is_raised_instead_of_becoming_evidence(self, get):
        get.return_value = response("<body>Access denied</body>")
        get.return_value.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        with self.assertRaises(ScrapeError) as error:
            invoke(scrape_url, {"url": URL})
        self.assertEqual(error.exception.url, URL)
        self.assertIn("403 Forbidden", str(error.exception))

    @patch("tools.requests.get", side_effect=requests.Timeout("read timed out"))
    def test_timeout_is_raised(self, get):
        with self.assertRaisesRegex(ScrapeError, "read timed out"):
            invoke(scrape_url, {"url": URL})

    @patch("tools.requests.get")
    def test_javascript_only_page_is_not_successful_evidence(self, get):
        get.return_value = response("<html><body><script>renderPage()</script></body></html>")
        with self.assertRaisesRegex(ScrapeError, "too little readable text"):
            invoke(scrape_url, {"url": URL})

    @patch("tools.requests.get")
    def test_pdf_is_rejected_instead_of_parsed_as_webpage_text(self, get):
        get.return_value = response("%PDF fake bytes", "application/pdf")
        with self.assertRaisesRegex(ScrapeError, "unsupported content type"):
            invoke(scrape_url, {"url": URL})

    @patch("tools.requests.get")
    def test_large_page_explicitly_records_excerpt_truncation(self, get):
        get.return_value = response(f"<body><p>{TEXT * 100}</p></body>")
        result = invoke(scrape_url, {"url": URL})
        self.assertTrue(result.artifact["truncated"])
        self.assertLessEqual(len(result.artifact["text"]), MAX_SCRAPED_CHARS)
        self.assertGreater(len(result.artifact["text"]), 3000)

    @patch("tools.requests.get")
    def test_scraping_does_not_require_search_api_key(self, get):
        get.return_value = response(f"<body>{TEXT}</body>")
        with patch.dict(os.environ, {}, clear=True):
            result = invoke(scrape_url, {"url": URL})
            self.assertEqual(result.artifact["text"], TEXT)
            with self.assertRaisesRegex(ToolException, "TAVILY_API_KEY"):
                invoke(web_search, {"query": "latency benchmark"})

    @patch("tools.requests.get")
    def test_non_http_url_is_rejected_before_fetch(self, get):
        with self.assertRaisesRegex(ScrapeError, "HTTP"):
            invoke(scrape_url, {"url": "file:///private.txt"})
        get.assert_not_called()


class SearchTests(unittest.TestCase):
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("tools.TavilyClient")
    def test_search_preserves_evidence_metadata_and_filters_bad_urls(self, client):
        client.return_value.search.return_value = {"results": [
            {"title": "Primary report", "url": URL, "content": TEXT * 10, "published_date": "2025-03-12"},
            {"title": "Duplicate", "url": URL, "content": "duplicate"},
            {"title": "Invalid", "url": "javascript:fake()", "content": "unusable"},
        ]}
        result = invoke(web_search, {"query": "latency benchmark"})
        self.assertEqual(len(result.artifact["sources"]), 1)
        source = result.artifact["sources"][0]
        self.assertEqual(source["url"], URL)
        self.assertEqual(source["published_at"], "2025-03-12")
        self.assertGreater(len(source["snippet"]), 300)
        self.assertIn(URL, result.content)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch("tools.TavilyClient")
    def test_empty_search_is_an_error(self, client):
        client.return_value.search.return_value = {"results": []}
        with self.assertRaisesRegex(ToolException, "no usable"):
            invoke(web_search, {"query": "latency benchmark"})


if __name__ == "__main__":
    unittest.main()
