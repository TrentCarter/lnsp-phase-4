import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("lnsp.api")

class TimingMiddleware(BaseHTTPMiddleware):
    """Logs request timing and basic request info for observability."""

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000

        log.info(
            "req=%s %s ms=%.2f status=%d",
            request.url.path,
            request.method,
            dt,
            response.status_code
        )

        return response