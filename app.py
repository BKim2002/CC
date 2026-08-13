"""Vercel이 탐지하는 FastAPI 애플리케이션 진입점."""

from chat.web import app

__all__ = ["app"]
