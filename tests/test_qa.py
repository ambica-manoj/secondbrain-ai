"""Grounding contract tests; no model download or Ollama server needed."""
import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.qa import MAX_CONTEXT_CHARS, NO_EVIDENCE, answer_question
from backend.llm import OllamaConnectionError, OllamaTimeoutError, OllamaResponseError
from backend.rag import RetrievalUnavailable


def hit(number=1, distance=0.1, text='Python is a programming language.'):
    return dict(chunk_id=f'doc:{number}', document_id='doc', filename='notes.pdf',
                page_number=number, text=text, distance=distance)


def test_grounded_answer_uses_only_cited_sources():
    store = Mock()
    store.search.return_value = [hit(), hit(2)]
    with patch('backend.qa.generate_response', return_value=json.dumps({'answer': 'Python is a language [2].'})) as generate:
        result = answer_question('What is Python?', store)
    assert result['sources'] == [{**hit(2), 'citation_id': 2}]
    prompt = generate.call_args.args[0]
    evidence = json.loads(prompt.split('Input JSON:\n')[1])
    assert evidence['question'] == 'What is Python?'
    assert evidence['evidence'][1]['id'] == 2
    assert generate.call_args.kwargs == {'grounded': True}


@pytest.mark.parametrize('hits', [[], [hit(distance=0.9)], [hit(distance=float('nan'))]])
def test_no_evidence_skips_llm(hits):
    store = Mock()
    store.search.return_value = hits
    with patch('backend.qa.generate_response') as generate:
        assert answer_question('question', store) == {'answer': NO_EVIDENCE, 'sources': []}
    generate.assert_not_called()


def test_context_is_bounded_and_returns_exact_passages():
    store = Mock()
    store.search.return_value = [hit(i, text='x' * 10000) for i in range(10)]
    with patch('backend.qa.generate_response', return_value='{"answer":"Evidence [4]"}') as generate:
        result = answer_question('question', store, top_k=10)
    evidence = json.loads(generate.call_args.args[0].split('Input JSON:\n')[1])['evidence']
    assert sum(len(item['text']) for item in evidence) == MAX_CONTEXT_CHARS
    assert result['sources'][0]['text'] == evidence[3]['text']


@pytest.mark.parametrize('raw', ['not json', '[]', '{}', '{"answer":""}',
                                 '{"answer":"No references"}', '{"answer":"Made up [99]"}',
                                 '{"answer":"Mixed [1] [0]"}'])
def test_invalid_model_answers_rejected(raw):
    store = Mock()
    store.search.return_value = [hit()]
    with patch('backend.qa.generate_response', return_value=raw):
        with pytest.raises(OllamaResponseError):
            answer_question('question', store)


def test_model_can_abstain_despite_close_match():
    store = Mock()
    store.search.return_value = [hit()]
    with patch('backend.qa.generate_response', return_value=json.dumps({'answer': NO_EVIDENCE})):
        assert answer_question('Unanswered question', store)['sources'] == []


@pytest.mark.parametrize('payload', [{}, {'question': ''}, {'question': ' '},
    {'question': 'x' * 2001}, {'question': 'x', 'top_k': 0},
    {'question': 'x', 'top_k': 11}, {'question': 'x', 'max_distance': -1},
    {'question': 'x', 'max_distance': 3}])
def test_ask_validation(payload):
    with TestClient(app) as client, patch('backend.main.store.search') as search:
        assert client.post('/ask', json=payload).status_code == 422
        search.assert_not_called()


@pytest.mark.parametrize('error,status', [(RetrievalUnavailable('index'), 503),
    (OllamaConnectionError('offline'), 503), (OllamaTimeoutError('timeout'), 504),
    (OllamaResponseError('bad response'), 502)])
def test_ask_error_mapping(error, status):
    with TestClient(app) as client, patch('backend.main.answer_question', side_effect=error):
        assert client.post('/ask', json={'question': 'Python?'}).status_code == status


def test_ask_api_retrieves_and_generates():
    with TestClient(app) as client, patch('backend.main.store.search', return_value=[hit()]) as search:
        with patch('backend.qa.generate_response', return_value='{"answer":"A language [1]."}'):
            response = client.post('/ask', json={'question': ' Python? ', 'top_k': 3})
    assert response.status_code == 200
    assert response.json()['sources'][0]['filename'] == 'notes.pdf'
    assert response.json()['sources'][0]['page_number'] == 1
    search.assert_called_once_with('Python?', 3)


def test_grounded_ollama_request_options():
    from backend.llm import generate_response
    from test_llm import FakeResponse
    with patch('backend.llm.urlopen', return_value=FakeResponse({'response': '{"answer":"Hi [1]"}'})) as urlopen:
        generate_response('evidence', grounded=True)
    payload = json.loads(urlopen.call_args.args[0].data)
    assert payload['format'] == 'json'
    assert payload['options'] == {'num_ctx': 4096, 'num_predict': 512, 'temperature': 0}


def test_upload_search_ask_with_real_index(tmp_path):
    from backend.rag import DocumentStore
    from test_retrieval import TinyEmbedder
    store = DocumentStore(tmp_path, TinyEmbedder())
    with TestClient(app) as client, patch('backend.main.store', store):
        assert client.post('/documents', files={'file': ('python.md', b'Python programming language')}).status_code == 200
        with patch('backend.qa.generate_response', return_value='{"answer":"A programming language [1]."}'):
            response = client.post('/ask', json={'question': 'Python'})
        assert response.status_code == 200
        source = response.json()['sources'][0]
        assert source['text'] == 'Python programming language'
        assert source['filename'] == 'python.md'
        assert source['page_number'] is None
