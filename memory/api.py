import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .llm import LLM, LLMError
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse
from .store import Store, Conflict


def create_app(store=None, llm=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store if store is not None else Store(os.getenv("MEMORY_DB", "data/memory.sqlite3"))
        app.state.llm = llm if llm is not None else LLM()
        yield

    app = FastAPI(title="CSIG Long-term Memory", version="0.1.0", lifespan=lifespan)

    @app.post("/add", response_model=AddResponse)
    def add(request: AddRequest):
        try:
            previous = app.state.store.existing(request)
            if previous:
                return previous
            graph = app.state.llm.extract(request)
            return app.state.store.add(request, graph)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        try:
            keywords = app.state.llm.keywords(request.query)
            return app.state.store.search(request, keywords)
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app

app = create_app()
