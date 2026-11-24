from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.config import MAX_UPLOAD_BYTES
from backend.ingestion import DocumentError
from backend.rag import RetrievalUnavailable, store

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


# Retrieval dependencies load lazily, so /health and /generate can start without
# downloading an embedding model. Run one backend worker on the initial 8 GB target.
class ImportResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    duplicate: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    text: str
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchHit]


@app.post('/documents', response_model=ImportResponse)
def import_document(file: UploadFile = File(...)):
    try:
        # Read at most the limit plus one byte; never copy an arbitrary upload into RAM.
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail='Documents must be at most 10 MiB.')
        return store.import_document(file.filename or '', content)
    except DocumentError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        file.file.close()


@app.post('/search', response_model=SearchResponse)
def search_documents(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail='Query must not be blank.')
    try:
        return {'results': store.search(request.query, request.top_k)}
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
