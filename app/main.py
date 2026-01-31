"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router


app = FastAPI(title="OncoAgent API")
app.include_router(router)

