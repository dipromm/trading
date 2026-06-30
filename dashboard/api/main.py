"""
MAS Trading System — FastAPI Backend.

Serves walk-forward results, holdout metrics, paper trading logs,
and experiment comparisons for the Next.js dashboard.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MAS Trading System API",
    description="Multi-Agent Trading System — Walk-Forward, Holdout & Paper Trading",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

from dashboard.api.routers import experiments, historical, live  # noqa: E402

app.include_router(historical.router)
app.include_router(live.router)
app.include_router(experiments.router)


@app.get("/health")
@limiter.limit("100/minute")
def health(request: Request) -> dict:
    return {"status": "ok"}
