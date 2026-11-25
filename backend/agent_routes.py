"""Study-plan previews and explicit approval of a stored proposal."""
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from backend.agent import preview_plan
from backend.database import Database, get_database
from backend.llm import OllamaConnectionError, OllamaResponseError, OllamaTimeoutError
from backend.models import InputModel
from backend.rag import store, RetrievalUnavailable
from backend.tools import save_approved_plan

router = APIRouter(prefix='/agent', tags=['Study planning'])
Storage = Annotated[Database, Depends(get_database)]


class PlanRequest(InputModel):
    request: str = Field(min_length=1, max_length=2000)


class Approval(InputModel):
    confirm: Literal[True]


@router.post('/plans', status_code=201)
def create_plan(request: PlanRequest, db: Storage):
    if not request.request.strip():
        raise HTTPException(422, 'Request must not be blank.')
    try:
        return preview_plan(request.request.strip(), store, db)
    except (RetrievalUnavailable, OllamaConnectionError) as error:
        raise HTTPException(503, str(error)) from error
    except OllamaTimeoutError as error:
        raise HTTPException(504, str(error)) from error
    except OllamaResponseError as error:
        raise HTTPException(502, str(error)) from error


@router.get('/plans/{plan_id}')
def get_plan(plan_id: UUID, db: Storage):
    return db.get_plan(str(plan_id))


@router.post('/plans/{plan_id}/approve')
def approve_plan(plan_id: UUID, request: Approval, db: Storage):
    return save_approved_plan(db, str(plan_id))
