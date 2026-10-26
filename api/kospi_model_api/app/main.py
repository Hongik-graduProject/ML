from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.schemas import HealthResponse
from app.services.artifact_store import ArtifactStore, ArtifactValidationError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = ArtifactStore(settings)
    app.state.settings = settings
    app.state.artifact_store = store
    app.state.startup_error = None
    try:
        store.load()
    except ArtifactValidationError as exc:
        app.state.startup_error = str(exc)
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "KOSPI 전체 기업 DART·가격·Macro DNA 기반 60거래일 "
        "KOSPI 대비 초과수익률 예측 API"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.include_router(router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
def health_live(request: Request) -> HealthResponse:
    store = request.app.state.artifact_store
    return HealthResponse(
        status="ok",
        api_version=request.app.state.settings.app_version,
        model_loaded=store.bundle is not None,
        predictions_loaded=not store.predictions.empty,
        detail=None,
    )


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
def health_ready(request: Request) -> JSONResponse | HealthResponse:
    store = request.app.state.artifact_store
    error = request.app.state.startup_error
    payload = HealthResponse(
        status="ready" if store.ready else "not_ready",
        api_version=request.app.state.settings.app_version,
        model_loaded=store.bundle is not None,
        predictions_loaded=not store.predictions.empty,
        detail=error,
    )
    if store.ready:
        return payload
    return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
