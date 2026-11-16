import os
import unittest
from unittest.mock import patch

from models import build_llm


class ModelConfigurationTests(unittest.TestCase):
    def test_missing_gemini_credentials_fail_before_model_request(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemini"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY or GEMINI_API_KEY"):
                build_llm()

    def test_invalid_provider_does_not_silently_use_ollama(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "unknown"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported MODEL_PROVIDER"):
                build_llm()

    def test_missing_groq_key_fails_before_model_request(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "groq"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                build_llm()

    @patch("langchain_groq.ChatGroq")
    def test_groq_uses_configured_model_and_key(self, groq):
        with patch.dict(os.environ, {
            "MODEL_PROVIDER": "groq", "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "openai/gpt-oss-120b",
        }, clear=True):
            build_llm()
        self.assertEqual(groq.call_args.kwargs["model"], "openai/gpt-oss-120b")
        self.assertEqual(groq.call_args.kwargs["groq_api_key"], "test-key")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_existing_gemini_key_name_is_supported(self, gemini):
        with patch.dict(os.environ, {
            "MODEL_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-3.6-flash",
        }, clear=True):
            build_llm()
        self.assertEqual(gemini.call_args.kwargs["google_api_key"], "test-key")
        self.assertEqual(gemini.call_args.kwargs["model"], "gemini-3.6-flash")


if __name__ == "__main__":
    unittest.main()
