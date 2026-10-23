
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.llm import (
    OLLAMA_MODEL,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    generate_response,
)

app = FastAPI(
    title="SecondBrain AI",
    description="A local-first personal knowledge assistant",
    version="0.1.0",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)


class GenerateResponse(BaseModel):
    model: str
    response: str


@app.get("/")
def home():
    return {
        "message": "Welcome to SecondBrain AI!",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt must not be blank.")

    try:
        response = generate_response(request.prompt)
    except OllamaTimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except OllamaConnectionError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaResponseError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return GenerateResponse(model=OLLAMA_MODEL, response=response)
