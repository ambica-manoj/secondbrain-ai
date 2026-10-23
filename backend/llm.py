import json
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))


class OllamaError(Exception):
    """Base error for Ollama request failures."""


class OllamaConnectionError(OllamaError):
    """Ollama could not be reached."""


class OllamaTimeoutError(OllamaError):
    """Ollama did not respond before the configured timeout."""


class OllamaResponseError(OllamaError):
    """Ollama returned an unsuccessful or invalid response."""


def generate_response(prompt: str) -> str:
    """Generate a response from the configured local Ollama model."""
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise OllamaResponseError(
            f"Ollama returned HTTP {error.code}: {detail or error.reason}"
        ) from error
    except (socket.timeout, TimeoutError) as error:
        raise OllamaTimeoutError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS:g} seconds."
        ) from error
    except URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise OllamaTimeoutError(
                f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS:g} seconds."
            ) from error
        raise OllamaConnectionError(
            f"Could not connect to Ollama at {OLLAMA_URL}: {error.reason}"
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OllamaResponseError("Ollama returned invalid JSON.") from error

    if not isinstance(result, dict):
        raise OllamaResponseError("Ollama returned an unexpected response format.")

    generated_text = result.get("response")
    if not isinstance(generated_text, str):
        raise OllamaResponseError("Ollama response did not contain generated text.")

    return generated_text
