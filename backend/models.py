"""Validated inputs and stored records for the local notes/tasks API."""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=20_000)]
TaskStatus = Literal['pending', 'in_progress', 'completed']


class InputModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class NoteCreate(InputModel):
    title: Title
    content: Content


class NoteUpdate(NoteCreate):
    """Replace the editable fields of a note."""


class TaskCreate(InputModel):
    title: Title
    description: Description = ''


class TaskUpdate(TaskCreate):
    status: TaskStatus


class Note(NoteCreate):
    id: str
    created_at: datetime
    updated_at: datetime


class Task(TaskCreate):
    id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
