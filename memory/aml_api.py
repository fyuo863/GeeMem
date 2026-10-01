"""Agent Memory Leaderboard synchronous Textual-track adapter."""
from contextlib import asynccontextmanager, contextmanager
import hmac
import json
import sqlite3
from threading import BoundedSemaphore
from typing import Literal
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field, field_validator
from .config import load_settings
from .llm import LLM, LLMError
from .models import StrictModel, AddRequest, SearchRequest
from .store import Store, Conflict


class AMLMessage(StrictModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=32000)
    timestamp: int | None = Field(default=None, ge=0, le=253402300799999, strict=True)
    speaker: str | None = Field(default=None, min_length=1, max_length=256, pattern=r'^\S(?:.*\S)?$')

    @field_validator('content')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Content must not be blank')
        return value


class Scope(StrictModel):
    user_id: str = Field(min_length=1, max_length=256)

    @field_validator('user_id')
    @classmethod
    def exact_id(cls, value):
        if value != value.strip():
            raise ValueError('IDs must not contain surrounding whitespace')
        return value


class AMLAdd(Scope):
    request_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=256)
    messages: list[AMLMessage] = Field(min_length=1, max_length=200)
    session_timestamp: int | None = Field(default=None, ge=0, le=253402300799999, strict=True)

    @field_validator('request_id','session_id')
    @classmethod
    def exact_ids(cls, value):
        return cls.exact_id(value)

    def internal(self):
        # Deterministic unknown-time sentinel preserves retry idempotency.
        return AddRequest(request_id=self.request_id,user_id=self.user_id,session_id=self.session_id,
            messages=[dict(role=m.role,content=m.content,timestamp=m.timestamp or 0) for m in self.messages])


class AMLSearch(Scope):
    query: str = Field(min_length=1, max_length=4000)
    options: list[str] | None = Field(default=None, max_length=100)
    top_k: int = Field(ge=1, le=100, strict=True)

    @field_validator('query')
    @classmethod
    def nonblank(cls, value):
        return AMLMessage.nonblank(value)

    @field_validator('options')
    @classmethod
    def valid_options(cls, values):
        if values is not None and any(not x.strip() or len(x)>4000 for x in values):
            raise ValueError('Options must be nonblank and at most 4000 characters each')
        return values


class AMLAddResponse(StrictModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class AMLHit(StrictModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    created_at: str | None = None


class AMLSearchResponse(StrictModel):
    data: list[AMLHit]


class EvaluationLLM(LLM):
    def __init__(self):
        super().__init__()
        if self.model != 'gpt-4o-mini':
            raise ValueError('AML open-source adapter requires LLM_MODEL=gpt-4o-mini')

    def complete(self, instruction, payload, schema):
        return super().complete(instruction + ' A source timestamp of 0 means unknown, '
            'not January 1970. Do not invent dates from missing timestamps.',payload,schema)


def create_app(store=None, llm=None, settings=None, backend=None):
    @asynccontextmanager
    async def lifespan(app):
        cfg=load_settings() if settings is None else dict(settings)
        mode=cfg.get('AML_AUTH_MODE','bearer').lower()
        key=cfg.get('AML_API_KEY','')
        if mode not in ('bearer','token','x-api-key','none'):
            raise ValueError('Unsupported AML_AUTH_MODE')
        if mode=='none' and cfg.get('AML_ALLOW_UNAUTHENTICATED','').lower()!='true':
            raise ValueError('Unauthenticated smoke requires AML_ALLOW_UNAUTHENTICATED=true')
        if mode!='none' and not key:
            raise ValueError('AML_API_KEY is required for authenticated evaluation')
        add_limit=int(cfg.get('AML_ADD_CONCURRENCY','1'));search_limit=int(cfg.get('AML_SEARCH_CONCURRENCY','4'))
        if not 1<=add_limit<=32 or not 1<=search_limit<=32:
            raise ValueError('AML concurrency must be between 1 and 32')
        app.state.auth_mode=mode;app.state.api_key=key
        app.state.add_slots=BoundedSemaphore(add_limit);app.state.search_slots=BoundedSemaphore(search_limit)
        mode_backend=cfg.get('MEMORY_BACKEND','vanilla')
        if mode_backend not in ('vanilla','graph'):
            raise ValueError('Unsupported MEMORY_BACKEND')
        app.state.backend=backend
        if backend is None and store is None and llm is None and mode_backend=='vanilla':
            from .vanilla import VanillaMemory
            app.state.backend=VanillaMemory(cfg)
        if app.state.backend is None:
            app.state.store=store if store is not None else Store(cfg.get('AML_MEMORY_DB','data/aml/memory.sqlite3'))
            app.state.llm=llm if llm is not None else EvaluationLLM()
        yield

    app=FastAPI(title='CSIG AML Textual Adapter',version='0.1.0',lifespan=lifespan)

    def authenticate(request: Request):
        mode=request.app.state.auth_mode
        if mode=='none':return
        if mode=='x-api-key':
            token=request.headers.get('x-api-key','')
        else:
            pieces=request.headers.get('authorization','').split(' ',1)
            token=pieces[1] if len(pieces)==2 and pieces[0].lower()==mode else ''
        if not hmac.compare_digest(token.encode(),request.app.state.api_key.encode()):
            raise HTTPException(401,'Invalid credentials')

    @contextmanager
    def capacity(semaphore):
        if not semaphore.acquire(blocking=False):
            raise HTTPException(429,'Capacity exhausted',headers={'Retry-After':'5'})
        try:yield
        finally:semaphore.release()

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo evaluation content/credentials inside validation responses.
        return JSONResponse(status_code=422,content={'detail':[
            dict(loc=e['loc'],type=e['type'],msg=e['msg']) for e in exc.errors()]})

    @app.exception_handler(sqlite3.OperationalError)
    async def storage_error(request, exc):
        return JSONResponse(status_code=503,content={'detail':'Storage temporarily unavailable'})

    @app.get('/health')
    def health():return {'status':'ok','track':'textual'}

    @app.post('/add',response_model=AMLAddResponse,dependencies=[Depends(authenticate)])
    def add(payload: AMLAdd):
        request=payload.internal()
        try:
            with capacity(app.state.add_slots):
                if app.state.backend is not None:
                    app.state.backend.add(payload)
                elif app.state.store.existing(request) is None:
                    graph=app.state.llm.extract(request)
                    app.state.store.add(request,graph)
            return AMLAddResponse(request_id=payload.request_id,user_id=payload.user_id,session_id=payload.session_id)
        except Conflict as exc:raise HTTPException(409,str(exc)) from exc
        except LLMError as exc:raise HTTPException(502,'Memory extraction failed') from exc
        except ValueError as exc:raise HTTPException(502,'Memory write failed') from exc

    @app.post('/search',response_model=AMLSearchResponse,response_model_exclude_none=True,
              dependencies=[Depends(authenticate)])
    def search(payload: AMLSearch):
        try:
            with capacity(app.state.search_slots):
                if app.state.backend is not None:
                    return app.state.backend.search(payload)
                # Options inform retrieval only; no answering, gold or benchmark-specific rules.
                query=payload.query
                if payload.options:
                    query+='\nMultiple-choice options (retrieval context only): '+json.dumps(payload.options,ensure_ascii=False)
                keywords=app.state.llm.keywords(query)
                request=SearchRequest(user_id=payload.user_id,query=payload.query,limit=payload.top_k)
                result=app.state.store.search(request,keywords)
                for hit in result['data']:
                    if hit.get('created_at')=='1970-01-01T00:00:00Z':hit.pop('created_at')
                return result
        except (LLMError, ValueError) as exc:raise HTTPException(502,'Memory retrieval failed') from exc

    return app


app=create_app()
