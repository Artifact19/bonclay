"""Shared helpers for routers: error shape and data paths."""
import os

from fastapi import HTTPException

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BACKEND_DIR, "data")


def api_error(status: int, error: str, code: str, details=None) -> HTTPException:
    """Build an HTTPException whose detail is {"error","code","details"}."""
    return HTTPException(status_code=status, detail={"error": error, "code": code, "details": details})
