from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from app.config import Settings
from app.services.artifact_store import ArtifactStore


def get_store(request: Request) -> ArtifactStore:
    return request.app.state.artifact_store


def get_runtime_settings(request: Request) -> Settings:
    return request.app.state.settings


def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if expected and (x_api_key is None or not secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않은 API key입니다.")
