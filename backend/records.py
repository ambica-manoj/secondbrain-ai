"""Local notes/tasks routes, ready for the future UI and validated agent tools."""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from backend.database import Database, get_database
from backend.models import Note, NoteCreate, NoteUpdate, Task, TaskCreate, TaskStatus, TaskUpdate

router = APIRouter(tags=['Notes and tasks'])
Storage = Annotated[Database, Depends(get_database)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


@router.post('/notes', response_model=Note, status_code=201)
def create_note(request: NoteCreate, db: Storage):
    return db.create('notes', request.model_dump())


@router.get('/notes', response_model=list[Note])
def list_notes(db: Storage, limit: Limit = 50, offset: Offset = 0):
    return db.list('notes', limit, offset)


@router.get('/notes/{record_id}', response_model=Note)
def get_note(record_id: UUID, db: Storage):
    return db.get('notes', str(record_id))


@router.put('/notes/{record_id}', response_model=Note)
def update_note(record_id: UUID, request: NoteUpdate, db: Storage):
    return db.update('notes', str(record_id), request.model_dump())


@router.delete('/notes/{record_id}', status_code=204)
def delete_note(record_id: UUID, db: Storage, confirm: bool = False):
    if not confirm:
        raise HTTPException(409, 'Deletion requires explicit confirmation: confirm=true.')
    db.delete('notes', str(record_id))
    return Response(status_code=204)


@router.post('/tasks', response_model=Task, status_code=201)
def create_task(request: TaskCreate, db: Storage):
    return db.create('tasks', request.model_dump())


@router.get('/tasks', response_model=list[Task])
def list_tasks(db: Storage, limit: Limit = 50, offset: Offset = 0, status: TaskStatus | None = None):
    return db.list('tasks', limit, offset, status)


@router.get('/tasks/{record_id}', response_model=Task)
def get_task(record_id: UUID, db: Storage):
    return db.get('tasks', str(record_id))


@router.put('/tasks/{record_id}', response_model=Task)
def update_task(record_id: UUID, request: TaskUpdate, db: Storage):
    return db.update('tasks', str(record_id), request.model_dump())


@router.delete('/tasks/{record_id}', status_code=204)
def delete_task(record_id: UUID, db: Storage, confirm: bool = False):
    if not confirm:
        raise HTTPException(409, 'Deletion requires explicit confirmation: confirm=true.')
    db.delete('tasks', str(record_id))
    return Response(status_code=204)
