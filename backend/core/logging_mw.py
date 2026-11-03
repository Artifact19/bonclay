"""Structured JSON request logging (structlog, falling back to stdlib logging)."""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, Request

from .errors import error_response

try:
    import structlog  # type: ignore
except ImportError:  # pragma: no cover
    structlog = None

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def _emit(event: str, **fields) -> None:
    if structlog is not None:
        structlog.get_logger("bonclay.http").info(event, **fields)
    else:
        logging.getLogger("bonclay.http").info(
            json.dumps({"event": event, **fields}, default=str))


def install_logging_middleware(app: FastAPI) -> None:
    """Add as the FIRST middleware so it sits inside CORS and error JSON still gets CORS headers."""
    setup_logging()

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        except Exception:  # noqa: BLE001
            logging.getLogger("bonclay.http").exception("unhandled error")
            return error_response("INTERNAL_ERROR", "Internal server error", 500,
                                  headers={"x-request-id": request_id})
        finally:
            _emit(
                "request",
                request_id=request_id,
                user_id=getattr(request.state, "user_id", None),
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
