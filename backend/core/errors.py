"""Uniform error handling: every error leaves the API as {"error", "code", "details"}."""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("bonclay.errors")

_STATUS_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


class AppError(Exception):
    """Application error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 400, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details


def error_body(code: str, message: str, details: Any = None) -> dict:
    # "detail" is kept as an alias for clients that still read FastAPI's default key.
    return {"error": message, "code": code, "details": details, "detail": message}


def error_response(code: str, message: str, status: int, details: Any = None,
                   headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=jsonable_encoder(error_body(code, message, details)),
                        headers=headers)


async def _app_error_handler(request: Request, exc: AppError):
    return error_response(exc.code, exc.message, exc.status, exc.details)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    code = _STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
    details = None
    if isinstance(detail, dict):
        code = detail.get("code", code)
        message = str(detail.get("error") or detail.get("message") or detail.get("detail") or code)
        details = detail.get("details")
    else:
        message = str(detail) if detail else code
    return error_response(code, message, exc.status_code, details,
                          headers=getattr(exc, "headers", None))


async def _validation_handler(request: Request, exc: RequestValidationError):
    errors = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
    fields = ", ".join(".".join(str(p) for p in e.get("loc", []) if p != "body") for e in errors[:5])
    msg = f"Invalid request: {fields}" if fields else "Invalid request"
    return error_response("VALIDATION_ERROR", msg, 422, errors)


async def _unhandled_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return error_response("INTERNAL_ERROR", "Internal server error", 500,
                          {"type": type(exc).__name__})


async def _rate_limit_handler(request: Request, exc: Exception):
    return error_response("RATE_LIMITED", "Rate limit exceeded. Please slow down.", 429,
                          {"limit": str(getattr(exc, "detail", ""))},
                          headers={"Retry-After": "60"})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
    try:
        from slowapi.errors import RateLimitExceeded
        app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    except ImportError:  # slowapi optional
        pass
