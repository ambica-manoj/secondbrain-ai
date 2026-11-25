"""Small SQLite repository; connections are owned and closed by each operation."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.config import DATA_DIR


class DatabaseUnavailable(RuntimeError):
    """The local structured storage could not be used."""


class RecordNotFound(LookupError):
    """A requested record does not exist."""


class Database:
    def __init__(self, path: Path = DATA_DIR / 'secondbrain.sqlite3'):
        self.path = Path(path)

    @contextmanager
    def _connection(self):
        connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5)
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        except (sqlite3.Error, OSError) as error:
            raise DatabaseUnavailable(
                'Could not access notes and tasks. Check the local data directory and try again.'
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def initialize(self):
        with self._connection() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'in_progress', 'completed')),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )''')

            connection.execute("""CREATE TABLE IF NOT EXISTS agent_plans (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, task_ids TEXT,
                created_at TEXT NOT NULL
            )""")

    def create_plan(self, payload):
        plan_id = str(uuid4())
        with self._connection() as connection:
            connection.execute('INSERT INTO agent_plans VALUES (?, ?, NULL, ?)',
                               (plan_id, json.dumps(payload), datetime.now(timezone.utc).isoformat()))
        return self.get_plan(plan_id)

    def get_plan(self, plan_id):
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM agent_plans WHERE id = ?', (plan_id,)).fetchone()
            if row is None:
                raise RecordNotFound('Plan not found.')
            return {'id': row['id'], **json.loads(row['payload']),
                    'status': 'saved' if row['task_ids'] is not None else 'preview',
                    'task_ids': json.loads(row['task_ids']) if row['task_ids'] is not None else []}

    def approve_plan(self, plan_id):
        with self._connection() as connection:
            # Serialize approvals across connections/processes; commit all tasks or none.
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT * FROM agent_plans WHERE id = ?', (plan_id,)).fetchone()
            if row is None:
                raise RecordNotFound('Plan not found.')
            if row['task_ids'] is not None:
                ids = json.loads(row['task_ids'])
            else:
                ids = []
                payload = json.loads(row['payload'])
                now = datetime.now(timezone.utc).isoformat()
                for task in payload['tasks']:
                    task_id = str(uuid4())
                    connection.execute('INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)',
                                       (task_id, task['title'], task['description'], 'pending', now, now))
                    ids.append(task_id)
                connection.execute('UPDATE agent_plans SET task_ids = ? WHERE id = ?',
                                   (json.dumps(ids), plan_id))
            records = []
            for task_id in ids:
                task = connection.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
                if task is None:
                    raise RecordNotFound('A previously saved task was deleted; this plan will not recreate it.')
                records.append(dict(task))
            return {'plan_id': plan_id, 'status': 'saved', 'tasks': records}

    @staticmethod
    def _table(kind):
        # Identifiers cannot use SQL parameters; accept only this fixed allowlist.
        if kind not in ('notes', 'tasks'):
            raise ValueError('Unknown record type.')
        return kind

    def create(self, kind, values):
        table = self._table(kind)
        now = datetime.now(timezone.utc).isoformat()
        record = {'id': str(uuid4()), 'title': values['title']}
        if table == 'notes':
            record['content'] = values['content']
        else:
            record.update(description=values.get('description', ''), status='pending')
        record.update(created_at=now, updated_at=now)
        columns = ', '.join(record)
        placeholders = ', '.join('?' for _ in record)
        with self._connection() as connection:
            connection.execute(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})',
                               tuple(record.values()))
            return dict(connection.execute(f'SELECT * FROM {table} WHERE id = ?',
                                           (record['id'],)).fetchone())

    def get(self, kind, record_id):
        table = self._table(kind)
        with self._connection() as connection:
            row = connection.execute(f'SELECT * FROM {table} WHERE id = ?', (record_id,)).fetchone()
            if row is None:
                raise RecordNotFound('Record not found.')
            return dict(row)

    def list(self, kind, limit=50, offset=0, status=None):
        table = self._table(kind)
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError('Invalid pagination.')
        where = ' WHERE status = ?' if table == 'tasks' and status is not None else ''
        parameters = [status] if where else []
        with self._connection() as connection:
            rows = connection.execute(
                f'SELECT * FROM {table}{where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?',
                (*parameters, limit, offset),
            ).fetchall()
            return [dict(row) for row in rows]

    def update(self, kind, record_id, values):
        table = self._table(kind)
        fields = ('title', 'content') if table == 'notes' else ('title', 'description', 'status')
        record = {field: values[field] for field in fields}
        record['updated_at'] = datetime.now(timezone.utc).isoformat()
        assignments = ', '.join(f'{field} = ?' for field in record)
        with self._connection() as connection:
            cursor = connection.execute(f'UPDATE {table} SET {assignments} WHERE id = ?',
                                        (*record.values(), record_id))
            if cursor.rowcount == 0:
                raise RecordNotFound('Record not found.')
            return dict(connection.execute(f'SELECT * FROM {table} WHERE id = ?',
                                           (record_id,)).fetchone())

    def delete(self, kind, record_id):
        table = self._table(kind)
        with self._connection() as connection:
            cursor = connection.execute(f'DELETE FROM {table} WHERE id = ?', (record_id,))
            if cursor.rowcount == 0:
                raise RecordNotFound('Record not found.')


database = Database()


def get_database():
    return database
