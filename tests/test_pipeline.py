import io
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import Mock, patch

import requests
from langchain_core.messages import AIMessage, ToolMessage

from pipeline import PipelineError, run_research_pipeline
from tools import ScrapeError


URLS = [f"https://source.example/report-{index}" for index in range(1, 4)]
TEXT = (
    "Measured results from March 2025: median latency was 42 ms over 1,200 requests "
    "on a single CPU. A timeout of 30 seconds was configured. This is a controlled "
    "measurement, not a forecast of future performance."
)


def report(url=URLS[0], extra=""):
    return f"Measured median latency was 42 ms in March 2025 [S1].\n{extra}\nSources:\n[S1] Report: {url}"


def message(name, artifact):
    return ToolMessage(
        name=name, tool_call_id="verified-test-call", content="Actual tool output",
        artifact=artifact, status="success",
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(redirect_stdout(io.StringIO()))
        self.context.enter_context(patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}))
        self.search = self.context.enter_context(patch("pipeline.build_search_agent")).return_value
        self.reader = self.context.enter_context(patch("pipeline.build_reader_agent")).return_value
        self.writer = self.context.enter_context(patch("pipeline.writer_chain"))
        self.critic = self.context.enter_context(patch("pipeline.critic_chain"))
        self.client = self.context.enter_context(patch("tools.TavilyClient")).return_value
        self.get = self.context.enter_context(patch("tools.requests.get"))
        self.get.side_effect = self.fetch
        self.candidates = [
            {"title": f"Primary report {index}", "url": url, "content": TEXT * 6}
            for index, url in enumerate(URLS, start=1)
        ]
        self.client.search.return_value = {"results": self.candidates}
        self.search.invoke.return_value = {"messages": [
            AIMessage(content="Here is a source I remember: https://invented.example/fake")
        ]}
        self.reader.invoke.return_value = {"messages": [
            AIMessage(content="I'll install requests for you. Here are generic 2020 projections.")
        ]}
        self.writer.invoke.return_value = report()
        self.critic.invoke.return_value = "Score: 8/10\nStrengths:\n- Grounded.\nAreas to Improve:\n- None."

    @staticmethod
    def fetch(url, **kwargs):
        response = Mock()
        response.url = url
        response.headers = {"Content-Type": "text/html"}
        response.content = f"<html><head><title>Measured report</title></head><body>{TEXT}</body></html>".encode()
        return response

    def test_hallucinated_tool_repair_is_replaced_by_real_webpage_evidence(self):
        state = run_research_pipeline("latency benchmark")
        self.client.search.assert_called_once()
        self.assertEqual({call.args[0] for call in self.get.call_args_list}, set(URLS))
        self.assertEqual(len(state["sources"]), 3)
        evidence = self.writer.invoke.call_args.args[0]["research"]
        self.assertIn(TEXT, evidence)
        self.assertNotIn("install requests", evidence)
        self.assertNotIn("2020 projections", evidence)
        self.assertNotIn("invented.example", evidence)
        self.assertEqual(self.critic.invoke.call_args.args[0]["research"], evidence)
        self.assertEqual(state["status"], "complete")

    def test_reader_receives_all_search_urls_after_long_snippets(self):
        run_research_pipeline("latency benchmark")
        reader_input = self.reader.invoke.call_args.args[0]["messages"][0][1]
        self.assertGreater(reader_input.index(URLS[-1]), 800)
        self.assertIn(URLS[-1], reader_input)

    def test_final_agent_prose_cannot_replace_successful_tool_text(self):
        artifact = {
            "url": URLS[0], "final_url": URLS[0], "title": "Primary report",
            "text": TEXT, "published_at": "2025-03-12", "retrieved_at": "2026-11-11T00:00:00+00:00",
            "truncated": False,
        }
        self.reader.invoke.return_value = {"messages": [
            message("scrape_url", artifact),
            AIMessage(content="Fabricated replacement summary with 2020 projections."),
        ]}
        state = run_research_pipeline("latency benchmark", max_sources=1)
        self.assertIn(TEXT, state["scraped_content"])
        self.assertNotIn("Fabricated replacement", state["scraped_content"])
        self.get.assert_not_called()

    def test_failed_first_source_falls_back_to_another_url(self):
        self.reader.invoke.side_effect = ScrapeError(URLS[0], "403 Forbidden")
        self.writer.invoke.return_value = report(URLS[1])
        state = run_research_pipeline("latency benchmark", max_sources=1)
        self.assertEqual(state["sources"][0]["url"], URLS[1])
        self.assertEqual(state["scrape_errors"][0]["url"], URLS[0])
        self.assertEqual([call.args[0] for call in self.get.call_args_list], [URLS[1]])

    def test_all_failed_sources_stop_before_writer_and_critic(self):
        self.get.side_effect = requests.Timeout("source unavailable")
        with self.assertRaisesRegex(PipelineError, "All sources failed"):
            run_research_pipeline("latency benchmark")
        self.writer.invoke.assert_not_called()
        self.critic.invoke.assert_not_called()

    def test_failed_tool_message_is_not_accepted_as_evidence(self):
        self.reader.invoke.return_value = {"messages": [
            ToolMessage(
                name="scrape_url", tool_call_id="failed-call", status="error",
                content="I could not fetch the page", artifact={"url": URLS[0], "text": "fake evidence"},
            )
        ]}
        state = run_research_pipeline("latency benchmark", max_sources=1)
        self.assertIn(TEXT, state["scraped_content"])
        self.assertNotIn("fake evidence", state["scraped_content"])
        self.get.assert_called_once()

    def test_revision_receives_previous_draft_critique_and_same_evidence(self):
        first_draft = report(extra="Unsupported citation [S99].")
        revised = report(extra="These values apply to the controlled test only [S1].")
        self.writer.invoke.side_effect = [first_draft, revised]
        self.critic.invoke.side_effect = ["Score: 4/10\nRemove unknown citations.", "Score: 8/10\nGrounded."]
        state = run_research_pipeline("latency benchmark", max_sources=1)
        first_input = self.writer.invoke.call_args_list[0].args[0]
        revision_input = self.writer.invoke.call_args_list[1].args[0]
        self.assertTrue(revision_input["draft"].startswith(first_draft.split("\nSources:")[0]))
        self.assertEqual(revision_input["research"], first_input["research"])
        self.assertIn("Score: 4/10", revision_input["feedback"])
        self.assertIn("Remove unknown citation [S99]", revision_input["feedback"])
        self.assertTrue(state["report"].startswith(revised.split("\nSources:")[0]))
        self.assertEqual(state["revisions"], 1)
        self.assertEqual(state["score"], 8)

    def test_fabricated_url_is_rejected_even_when_critic_gives_high_score(self):
        self.writer.invoke.return_value = report(extra="Claim [S99]: https://invented.example/fake")
        with self.assertRaisesRegex(PipelineError, "citation validation"):
            run_research_pipeline("latency benchmark", max_sources=1, max_revisions=0)

    def test_bibliography_is_built_from_verified_urls_without_extra_model_call(self):
        self.writer.invoke.return_value = (
            "Median latency was 42 ms [S1].\nSources:\n[S1] https://invented.example/fake"
        )
        state = run_research_pipeline("latency benchmark", max_sources=1, max_revisions=0)
        self.assertIn(URLS[0], state["report"])
        self.assertNotIn("invented.example", state["report"])
        self.writer.invoke.assert_called_once()

    def test_source_list_alone_does_not_count_as_inline_citations(self):
        self.writer.invoke.return_value = (
            f"Median latency was 42 ms.\n**Sources**\n[S1] Report: {URLS[0]}"
        )
        with self.assertRaisesRegex(PipelineError, "inline source-ID citations"):
            run_research_pipeline("latency benchmark", max_sources=1, max_revisions=0)

    def test_unknown_source_in_grouped_citation_is_rejected(self):
        self.writer.invoke.return_value = report(extra="Another claim [S1, S99].")
        with self.assertRaisesRegex(PipelineError, "unknown citation"):
            run_research_pipeline("latency benchmark", max_sources=1, max_revisions=0)

    def test_bold_score_label_is_parsed(self):
        self.critic.invoke.return_value = "**Score:** 8/10"
        state = run_research_pipeline("latency benchmark", max_sources=1, max_revisions=0)
        self.assertEqual(state["score"], 8)

    def test_low_score_after_revision_limit_is_reported_as_needing_review(self):
        self.critic.invoke.return_value = "Score: 4/10\nMore analytical depth is needed."
        state = run_research_pipeline("latency benchmark", max_sources=1, max_revisions=0)
        self.assertEqual(state["status"], "needs_review")
        self.assertEqual(state["score"], 4)

    def test_search_failure_stops_before_reader_or_writer(self):
        self.client.search.return_value = {"results": []}
        with self.assertRaisesRegex(PipelineError, "Search failed"):
            run_research_pipeline("latency benchmark")
        self.reader.invoke.assert_not_called()
        self.writer.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
