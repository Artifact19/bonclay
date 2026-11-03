"""asyncio.Queue-backed background job runner (single worker: TTS is GPU/CPU bound)."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from .errors import AppError

log = logging.getLogger("bonclay.jobs")
MAX_JOBS = 500


class JobRunner:
    def __init__(self, workers: int = 1):
        self.workers = workers
        self.jobs: Dict[str, dict] = {}
        self._queue: Optional[asyncio.Queue] = None
        self._tasks: list = []

    def start(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        self._tasks = [t for t in self._tasks if not t.done()]
        while len(self._tasks) < self.workers:
            self._tasks.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks = []

    def submit(self, user_id: str, factory: Callable[[], Awaitable[Any]], kind: str = "generate") -> str:
        self.start()
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id, "user_id": user_id, "kind": kind, "status": "queued",
            "created_at": time.time(), "started_at": None, "finished_at": None,
            "result": None, "error": None, "progress": None,
        }
        self._queue.put_nowait((job_id, factory))
        self._prune()
        return job_id

    def set_progress(self, job_id: str, progress: Any) -> None:
        """A running job reports how far it is (shown by get(); e.g. {"done": 12, "total": 60})."""
        job = self.jobs.get(job_id)
        if job is not None:
            job["progress"] = progress

    def get(self, job_id: str, user_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job or job["user_id"] != user_id:
            raise AppError("JOB_NOT_FOUND", "Job not found", 404)
        return {k: v for k, v in job.items() if k != "user_id"}

    async def _worker(self) -> None:
        while True:
            job_id, factory = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            job["status"], job["started_at"] = "running", time.time()
            try:
                job["result"] = await factory()
                job["status"] = "done"
            except AppError as exc:
                job["status"], job["error"] = "failed", {"code": exc.code, "message": exc.message,
                                                          "details": exc.details}
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("job %s failed", job_id)
                job["status"], job["error"] = "failed", {"code": "INTERNAL_ERROR",
                                                          "message": "Job failed", "details": None}
            finally:
                job["finished_at"] = time.time()

    def _prune(self) -> None:
        if len(self.jobs) <= MAX_JOBS:
            return
        finished = sorted((j for j in self.jobs.values() if j["status"] in ("done", "failed")),
                          key=lambda j: j["created_at"])
        for j in finished[: len(self.jobs) - MAX_JOBS]:
            self.jobs.pop(j["id"], None)


job_runner = JobRunner()
