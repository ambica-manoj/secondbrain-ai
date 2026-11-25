import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from backend import agent, agent_routes
from backend.database import Database, DatabaseUnavailable, RecordNotFound, get_database
from backend.llm import OllamaResponseError, OllamaTimeoutError
from backend.main import app

@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / 'test.sqlite3')
    db.initialize()
    return db

@pytest.fixture
def store():
    return Mock(search=Mock(return_value=[dict(text='Regularization reduces overfitting.', distance=.2,
        filename='notes.md', page_number=None, document_id='doc', chunk_id='chunk')]))

@pytest.fixture
def generate(monkeypatch):
    mock = Mock(side_effect=[json.dumps(dict(tool='search_documents', query='overfitting')),
        json.dumps(dict(title='Revision', tasks=[dict(title='Session 1', description='Review regularization.', source_ids=[1])]))])
    monkeypatch.setattr(agent, 'generate_response', mock)
    return mock

def test_persistent_preview_and_concurrent_approval(db, store, generate):
    plan = agent.preview_plan('Make a plan', store, db)
    assert plan['status'] == 'preview'
    assert db.list('tasks') == []
    assert Database(db.path).get_plan(plan['id']) == plan
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: Database(db.path).approve_plan(plan['id']), range(4)))
    assert all(r == results[0] for r in results)
    assert len(db.list('tasks')) == 1
    assert db.get_plan(plan['id'])['status'] == 'saved'
    store.search.assert_called_once_with('overfitting', 4)

def test_empty_evidence(db, store, generate):
    store.search.return_value = []
    assert agent.preview_plan('Make a plan', store, db)['status'] == 'insufficient_evidence'
    assert generate.call_count == 1
    assert db.list('tasks') == []

@pytest.mark.parametrize('raw', ['not json', '{"tool":"shell","query":"rm"}', '{"tool":"search_documents","query":""}'])
def test_invalid_tool(db, store, monkeypatch, raw):
    monkeypatch.setattr(agent, 'generate_response', lambda *a, **k: raw)
    with pytest.raises(OllamaResponseError):
        agent.preview_plan('Make a plan', store, db)
    store.search.assert_not_called()

def test_invalid_plan(db, store, generate):
    generate.side_effect = [json.dumps(dict(tool='search_documents', query='topic')),
        json.dumps(dict(title='Plan', tasks=[dict(title='Task', source_ids=[99])]))]
    with pytest.raises(OllamaResponseError):
        agent.preview_plan('Make a plan', store, db)
    assert db.list('tasks') == []

def test_atomic_rollback(db):
    plan = db.create_plan(dict(tasks=[dict(title='Valid', description=''), dict(title=None, description='')]))
    with pytest.raises(DatabaseUnavailable):
        db.approve_plan(plan['id'])
    assert db.list('tasks') == []
    assert db.get_plan(plan['id'])['status'] == 'preview'

def test_deleted_task_not_recreated(db, store, generate):
    plan = agent.preview_plan('Make a plan', store, db)
    saved = db.approve_plan(plan['id'])
    db.delete('tasks', saved['tasks'][0]['id'])
    with pytest.raises(RecordNotFound):
        db.approve_plan(plan['id'])
    assert db.list('tasks') == []

def test_api(db, store, generate, monkeypatch):
    app.dependency_overrides[get_database] = lambda: db
    monkeypatch.setattr(agent_routes, 'store', store)
    try:
        with TestClient(app) as client:
            assert client.post('/agent/plans', json={'request':' '}).status_code == 422
            response = client.post('/agent/plans', json={'request':'Make a plan'})
            assert response.status_code == 201
            url = '/agent/plans/' + response.json()['id']
            assert client.get(url).json()['status'] == 'preview'
            for body in ({}, {'confirm':False}):
                assert client.post(url+'/approve', json=body).status_code == 422
            assert db.list('tasks') == []
            assert client.post(url+'/approve', json={'confirm':True}).status_code == 200
            assert len(db.list('tasks')) == 1
            generate.side_effect = OllamaTimeoutError('Timed out')
            assert client.post('/agent/plans', json={'request':'Make a plan'}).status_code == 504
    finally:
        app.dependency_overrides.clear()

@pytest.mark.parametrize('raw', ['not json', '{"title":"Plan","tasks":[{"title":"Task","source_ids":["1"]}]}', '{"title":"Plan","tasks":[{"title":"Task","source_ids":[]}]}'])
def test_malformed_plan(db, store, generate, raw):
    generate.side_effect = [json.dumps(dict(tool='search_documents', query='topic')), raw]
    with pytest.raises(OllamaResponseError):
        agent.preview_plan('Make a plan', store, db)
    assert db.list('tasks') == []

def test_model_abstention(db, store, generate):
    generate.side_effect = [json.dumps(dict(tool='search_documents', query='topic')),
                           json.dumps(dict(title='Insufficient evidence', tasks=[]))]
    assert agent.preview_plan('Make a plan', store, db)['status'] == 'insufficient_evidence'
    assert db.list('tasks') == []
