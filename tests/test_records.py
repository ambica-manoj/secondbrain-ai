"""Persistent record lifecycle and API contracts; no inference needed."""
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.database import Database, DatabaseUnavailable, get_database
from backend.main import app


@pytest.fixture
def client(tmp_path):
    db = Database(tmp_path / 'nested' / 'secondbrain.sqlite3')
    app.dependency_overrides[get_database] = lambda: db
    with patch('backend.main.get_database', return_value=db):
        with TestClient(app) as test_client:
            yield test_client
    app.dependency_overrides.clear()


@pytest.mark.parametrize('kind,body,updated', [
    ('notes', {'title': ' Revision ', 'content': ' Read about Python '},
     {'title': 'SQL', 'content': 'Review joins'}),
    ('tasks', {'title': ' Revision ', 'description': ' Read about Python '},
     {'title': 'SQL', 'description': 'Review joins', 'status': 'completed'}),
])
def test_record_lifecycle(client, kind, body, updated):
    assert client.get(f'/{kind}').json() == []
    response = client.post(f'/{kind}', json=body)
    assert response.status_code == 201
    record = response.json()
    assert record['title'] == 'Revision'
    assert record['created_at'].endswith('Z')
    if kind == 'tasks':
        assert record['status'] == 'pending'
    url = f'/{kind}/{record["id"]}'
    assert client.get(url).json() == record
    assert client.get(f'/{kind}').json() == [record]
    changed = client.put(url, json=updated)
    assert changed.status_code == 200
    assert changed.json()['created_at'] == record['created_at']
    assert changed.json()['updated_at'] >= record['updated_at']
    for key, value in updated.items():
        assert changed.json()[key] == value
    for query in ('', '?confirm=false'):
        assert client.delete(url + query).status_code == 409
        assert client.get(url).status_code == 200
    assert client.delete(url + '?confirm=true').status_code == 204
    assert client.get(url).status_code == 404
    assert client.put(url, json=updated).status_code == 404
    assert client.delete(url + '?confirm=true').status_code == 404


def test_reopen_storage_retains_records(tmp_path):
    path = tmp_path / 'data' / 'records.sqlite3'
    first = Database(path)
    first.initialize()
    # SQL metacharacters and Unicode are stored literally.
    title = "Robert'); DROP TABLE notes; -- 📚"
    note = first.create('notes', {'title': title, 'content': '私のノート'})
    task = first.create('tasks', {'title': 'Revise'})
    second = Database(path)
    second.initialize()
    assert second.get('notes', note['id']) == note
    assert second.get('tasks', task['id']) == task
    assert second.list('notes') == [note]


@pytest.mark.parametrize('kind,payload', [
    ('notes', {'title': ' ', 'content': 'valid'}),
    ('notes', {'title': 'valid', 'content': '\n'}),
    ('notes', {'title': 'x' * 201, 'content': 'valid'}),
    ('notes', {'title': 'valid', 'content': 'x' * 20001}),
    ('notes', {'title': 'valid'}),
    ('notes', {'title': 'valid', 'content': 'valid', 'id': 'custom'}),
    ('tasks', {'title': ''}),
    ('tasks', {'title': 'x' * 201}),
    ('tasks', {'title': 'valid', 'description': 'x' * 20001}),
    ('tasks', {'title': 'valid', 'description': None}),
    ('tasks', {'title': 'valid', 'status': 'completed'}),
])
def test_invalid_create_does_not_write(client, kind, payload):
    assert client.post(f'/{kind}', json=payload).status_code == 422
    assert client.get(f'/{kind}').json() == []


def test_task_status_filter_and_pagination(client):
    tasks = [client.post('/tasks', json={'title': f'Task {i}'}).json() for i in range(3)]
    task = tasks[0]
    client.put(f'/tasks/{task["id"]}', json={
        'title': task['title'], 'description': '', 'status': 'in_progress',
    })
    assert len(client.get('/tasks?status=pending').json()) == 2
    assert client.get('/tasks?status=in_progress').json()[0]['id'] == task['id']
    assert client.get('/tasks?limit=1&offset=1').json()[0]['id'] == tasks[1]['id']
    assert client.get('/tasks?offset=20').json() == []
    for query in ('limit=0', 'limit=101', 'offset=-1', 'status=unknown'):
        assert client.get('/tasks?' + query).status_code == 422
    assert client.put(f'/tasks/{task["id"]}', json={
        'title': 'bad', 'description': '', 'status': 'unknown',
    }).status_code == 422
    assert client.get(f'/tasks/{task["id"]}').json()['title'] == task['title']


@pytest.mark.parametrize('kind', ['notes', 'tasks'])
def test_missing_and_invalid_ids(client, kind):
    assert client.get(f'/{kind}/{uuid4()}').status_code == 404
    assert client.get(f'/{kind}/invalid').status_code == 422


def test_database_failure_returns_service_unavailable(client):
    with patch.object(Database, 'list', side_effect=DatabaseUnavailable('Storage unavailable.')):
        response = client.get('/notes')
    assert response.status_code == 503
    assert response.json() == {'detail': 'Storage unavailable.'}


def test_unwritable_database_reports_clear_error(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    with pytest.raises(DatabaseUnavailable):
        Database(blocked / 'records.sqlite3').initialize()


def test_unknown_tables_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        Database(tmp_path / 'db').list('notes; DROP TABLE tasks')
