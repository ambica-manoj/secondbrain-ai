"""Persistent semantic retrieval. Answer generation is a later milestone."""
import hashlib
import os
from pathlib import Path
from threading import RLock

from backend.config import DATA_DIR, EMBEDDING_MODEL
from backend.ingestion import DocumentError, chunk_pages, parse_document

# Set before importing Chroma; document content stays local.
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')


class RetrievalUnavailable(RuntimeError):
    """The local embedding model or index could not be used."""


class DocumentStore:
    def __init__(self, data_dir: Path = DATA_DIR, embedder=None):
        self.data_dir = Path(data_dir)
        self._embedder = embedder
        self._collection = None
        self._client = None
        self._lock = RLock()

    def _index(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.data_dir / 'chroma'),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                'documents_minilm_v1',
                metadata={'hnsw:space': 'cosine', 'embedding_model': EMBEDDING_MODEL},
                embedding_function=None,
            )
        return self._collection

    def _model(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            # CPU avoids competing with Ollama for scarce GPU memory.
            self._embedder = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        return self._embedder

    def import_document(self, filename: str, content: bytes) -> dict:
        name, pages = parse_document(filename, content)
        document_id = hashlib.sha256(content).hexdigest()
        with self._lock:
            try:
                collection = self._index()
                existing = collection.get(where={'document_id': document_id}, include=['metadatas'])
                if existing['ids']:
                    return {'document_id': document_id,
                            'filename': existing['metadatas'][0]['filename'],
                            'chunk_count': len(existing['ids']), 'duplicate': True}
                model = self._model()
                chunks = chunk_pages(pages, model.tokenizer)
                vectors = model.encode([c['text'] for c in chunks],
                                       normalize_embeddings=True, batch_size=16).tolist()
                ids = [f'{document_id}:{i}' for i in range(len(chunks))]
                metadata = [{'document_id': document_id, 'filename': name,
                             'page_number': c['page_number'], 'chunk_index': i}
                            for i, c in enumerate(chunks)]
                # One bounded write, with cleanup if Chroma reports a write failure.
                try:
                    collection.add(ids=ids, documents=[c['text'] for c in chunks],
                                   embeddings=vectors, metadatas=metadata)
                except Exception:
                    collection.delete(where={'document_id': document_id})
                    raise
                return {'document_id': document_id, 'filename': name,
                        'chunk_count': len(chunks), 'duplicate': False}
            except DocumentError:
                raise
            except Exception as error:
                raise RetrievalUnavailable(
                    'Could not index the document. Check local model availability and data directory permissions.'
                ) from error

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not query.strip() or not 1 <= top_k <= 10:
            raise ValueError('Provide a nonblank query and top_k between 1 and 10.')
        with self._lock:
            try:
                collection = self._index()
                count = collection.count()
                if count == 0:
                    return []
                vector = self._model().encode([query], normalize_embeddings=True).tolist()
                results = collection.query(query_embeddings=vector, n_results=min(top_k, count),
                                           include=['documents', 'metadatas', 'distances'])
                return [{'chunk_id': chunk_id, 'text': text,
                         'document_id': metadata['document_id'], 'filename': metadata['filename'],
                         'page_number': metadata['page_number'] or None,
                         'distance': distance}
                        for chunk_id, text, metadata, distance in zip(
                            results['ids'][0], results['documents'][0],
                            results['metadatas'][0], results['distances'][0])]
            except Exception as error:
                raise RetrievalUnavailable('Could not search the local document index.') from error


store = DocumentStore()
