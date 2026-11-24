"""Real persistent Chroma index with tiny deterministic embeddings; no downloads."""
import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend.ingestion import DocumentError, Page, chunk_pages, parse_document
from backend.main import app
from backend.rag import DocumentStore, RetrievalUnavailable


class WordTokenizer:
    def __init__(self):
        self.words = {}

    def encode(self, text, **kwargs):
        tokens = text.split()
        for token in tokens:
            self.words.setdefault(token, len(self.words) + 1)
        return [self.words[token] for token in tokens]

    def decode(self, tokens, **kwargs):
        inverse = {v: k for k, v in self.words.items()}
        return ' '.join(inverse[token] for token in tokens)


class TinyEmbedder:
    tokenizer = WordTokenizer()

    def encode(self, texts, **kwargs):
        return np.array([[1 + text.lower().count('python'),
                          1 + text.lower().count('cloud'), 1] for text in texts], dtype=float)


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path, TinyEmbedder())


@pytest.fixture
def client(store):
    with patch('backend.main.store', store):
        with TestClient(app) as client:
            yield client


def test_import_search_and_restart(store):
    first = store.import_document('../python.md', b'Python Python Python programming language')
    store.import_document('cloud.md', b'Cloud Cloud Cloud infrastructure')
    result = store.search('Python Python')
    assert result[0]['filename'] == 'python.md'
    assert result[0]['page_number'] is None
    assert result[0]['document_id'] == first['document_id']
    assert result[0]['chunk_id'].startswith(first['document_id'])
    restarted = DocumentStore(store.data_dir, TinyEmbedder())
    assert restarted.search('Python Python') == result


def test_duplicate_import_is_idempotent(store):
    first = store.import_document('notes.md', b'Python notes')
    second = store.import_document('renamed.md', b'Python notes')
    assert second == {**first, 'duplicate': True}
    assert store._index().count() == first['chunk_count']


def test_empty_index_does_not_load_embedding_model(tmp_path):
    store = DocumentStore(tmp_path)
    with patch.object(store, '_model', side_effect=AssertionError('Should not load')):
        assert store.search('Python') == []


def test_token_chunks_overlap_and_keep_pages():
    tokenizer = WordTokenizer()
    chunks = chunk_pages([Page('a b c d e f', 2), Page('g h', 3)], tokenizer, 4, 1)
    assert chunks == [{'text': 'a b c d', 'page_number': 2},
                      {'text': 'd e f', 'page_number': 2},
                      {'text': 'g h', 'page_number': 3}]


@pytest.mark.parametrize('name,content', [
    ('notes.txt', b'hello'), ('notes.md', b''), ('notes.md', b'  \n'),
    ('notes.md', b'\xff'), ('bad.pdf', b'not a pdf'),
])
def test_invalid_documents(name, content):
    with pytest.raises(DocumentError):
        parse_document(name, content)


def test_scanned_pdf_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(DocumentError, match='OCR'):
        parse_document('blank.pdf', output.getvalue())


def test_pdf_page_metadata():
    # Small real PDF built using pypdf's stream API, without fixture binaries.
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    for text in ['Python programming', 'Cloud computing']:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 20 200 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    name, pages = parse_document('notes.pdf', output.getvalue())
    assert name == 'notes.pdf'
    assert [p.number for p in pages] == [1, 2]
    assert 'Cloud' in pages[1].text


def test_upload_and_search_api(client):
    response = client.post('/documents', files={'file': ('notes.md', b'Python Python notes')})
    assert response.status_code == 200
    assert response.json()['chunk_count'] == 1
    response = client.post('/search', json={'query': 'Python', 'top_k': 3})
    assert response.status_code == 200
    assert response.json()['results'][0]['filename'] == 'notes.md'


@pytest.mark.parametrize('payload', [{'query': ' '}, {'query': ''},
                                     {'query': 'x', 'top_k': 0}, {'query': 'x', 'top_k': 11}])
def test_invalid_search_api(client, payload):
    assert client.post('/search', json=payload).status_code == 422


def test_oversized_upload(client):
    with patch('backend.main.MAX_UPLOAD_BYTES', 8):
        assert client.post('/documents', files={'file': ('x.md', b'123456789')}).status_code == 413


def test_unavailable_index_api(client):
    with patch('backend.main.store.search', side_effect=RetrievalUnavailable('Unavailable')):
        assert client.post('/search', json={'query': 'Python'}).status_code == 503


def test_embedding_failure_leaves_no_chunks(store):
    with patch.object(store._embedder, 'encode', side_effect=RuntimeError('model unavailable')):
        with pytest.raises(RetrievalUnavailable):
            store.import_document('notes.md', b'Python notes')
    assert store._index().count() == 0


def test_existing_generation_api(client):
    with patch('backend.main.generate_response', return_value='Hello'):
        response = client.post('/generate', json={'prompt': 'Say hello'})
    assert response.status_code == 200
    assert response.json()['response'] == 'Hello'
    assert client.post('/generate', json={'prompt': ' '}).status_code == 422


def test_index_survives_new_python_process(store):
    import json
    import os
    import subprocess
    import sys
    store.import_document('persistent.md', b'Python notes')
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_retrieval import TinyEmbedder
from backend.rag import DocumentStore
print(json.dumps(DocumentStore(Path(sys.argv[1]), TinyEmbedder()).search('Python')))
'''
    completed = subprocess.run([sys.executable, '-c', script, str(store.data_dir)],
                               capture_output=True, text=True, check=True,
                               env=os.environ.copy(), timeout=30)
    assert json.loads(completed.stdout)[0]['filename'] == 'persistent.md'


def test_failed_index_write_rolls_back(store):
    collection = store._index()
    original_add = collection.add

    def fail_after_write(**kwargs):
        original_add(**kwargs)
        raise RuntimeError('simulated write failure')

    with patch.object(collection, 'add', side_effect=fail_after_write):
        with pytest.raises(RetrievalUnavailable):
            store.import_document('notes.md', b'Python notes')
    assert collection.count() == 0
    assert not store.import_document('notes.md', b'Python notes')['duplicate']


def test_unsupported_upload_api(client):
    assert client.post('/documents', files={'file': ('x.exe', b'content')}).status_code == 422


def test_upload_model_failure_api(client):
    with patch('backend.main.store.import_document', side_effect=RetrievalUnavailable('Unavailable')):
        assert client.post('/documents', files={'file': ('x.md', b'content')}).status_code == 503
