import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from backend.llm import (
    OllamaConnectionError,
    OllamaResponseError,
    generate_response,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class GenerateResponseTests(unittest.TestCase):
    @patch("backend.llm.urlopen")
    def test_sends_non_streaming_request_and_returns_generated_text(self, urlopen):
        urlopen.return_value = FakeResponse({"response": "Hello from Ollama"})

        result = generate_response("Say hello")

        self.assertEqual(result, "Hello from Ollama")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(
            json.loads(request.data),
            {"model": "qwen2.5:3b", "prompt": "Say hello", "stream": False},
        )

    @patch("backend.llm.urlopen", side_effect=URLError("connection refused"))
    def test_reports_unavailable_ollama(self, _urlopen):
        with self.assertRaises(OllamaConnectionError):
            generate_response("Say hello")

    @patch("backend.llm.urlopen")
    def test_rejects_response_without_generated_text(self, urlopen):
        urlopen.return_value = FakeResponse({"done": True})

        with self.assertRaises(OllamaResponseError):
            generate_response("Say hello")

    @patch("backend.llm.urlopen")
    def test_rejects_invalid_json(self, urlopen):
        urlopen.return_value = FakeResponse(b"not json")

        with self.assertRaises(OllamaResponseError):
            generate_response("Say hello")


if __name__ == "__main__":
    unittest.main()
