"""The FastAPI app factory: the static bundle, the guards, the error envelope.

The skeleton T014 asks for, and nothing more:

* FR-013c — the bind is asserted at startup and every request's ``Host`` and
  ``Origin`` are checked before it reaches a route.
* FR-013a/FR-013b — the configured static root is mounted verbatim. The backend
  composes no markup: it either serves the bundle's own files or refuses.
* FR-086 — the static root, the port and every artifact root come from the
  passed-in :class:`~manga_text_seg.config.WebConfig`. Nothing here is derived
  or hardcoded.
* FR-079 — every refusal leaves through :func:`~manga_text_seg.web.errors.error_response`,
  so the envelope shape and the status codes live in one place.

The routers land in T024+; they mount on this app, so the factory is the single
place that knows about them.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import WebConfig
from .demo import register_demo_routes
from .errors import error_response
from .security import (
    ArtifactRefused,
    assert_loopback_bind,
    host_allowed,
    origin_allowed,
)

#: The health route. It has no FR of its own — plan.md's Phase 0 asks for "a
#: health endpoint" and leaves the path open, so it sits under the same /api
#: prefix as everything else the backend answers.
HEALTH_PATH = "/api/health"


def _static_ready(root: Path) -> bool:
    """Whether the configured static root holds a bundle to serve (FR-013a).

    An empty directory is the same condition as a missing one: mounting it
    would answer 404 for the UI while looking, from the outside, exactly like a
    served bundle.
    """
    return root.is_dir() and any(root.iterdir())


def _not_ready(root: Path) -> JSONResponse:
    """The refusal `/` returns while there is no bundle, naming the expected root."""
    return error_response(
        "not_ready",
        f"the static root '{root}' holds no frontend bundle; drop one there "
        f"and reload (FR-013a)",
        {"static_root": str(root)},
    )


def create_app(config: WebConfig) -> FastAPI:
    """Build the Spec 005 app for ``config`` (FR-086).

    ``assert_loopback_bind`` runs first and raises
    :class:`~manga_text_seg.web.security.NonLoopbackBind`, so a non-loopback
    bind is refused at startup rather than on the first request (FR-013c).
    """
    assert_loopback_bind(config)

    app = FastAPI(title="manga-text-seg web demo", docs_url=None, redoc_url=None)
    app.state.config = config

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        """Refuse a request that did not come from this loopback service.

        A browser cannot set these headers to evade the check — it is the
        browser that sets them — so this is the DNS-rebinding guard FR-013c
        asks for, not a CSRF token.
        """
        if not host_allowed(config, request.headers.get("host")):
            return error_response(
                "invalid_request", "the Host header is not this service (FR-013c)"
            )
        if not origin_allowed(config, request.headers.get("origin")):
            return error_response(
                "invalid_request",
                "the Origin header is not this service (FR-013c)",
            )
        return await call_next(request)

    # --- error handlers (FR-079) -------------------------------------------
    # One envelope for every failure. Handlers rather than try/except in each
    # route: a route that forgets is exactly the one that leaks a stack trace.

    @app.exception_handler(ArtifactRefused)
    async def _artifact_refused(request: Request, exc: ArtifactRefused):
        return error_response("invalid_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, exc: RequestValidationError):
        return error_response(
            "invalid_request", "the request did not match the endpoint's contract"
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        # Starlette raises these for its own reasons — an unmatched path, a
        # method the route does not accept. Both are the caller's mistake.
        category = "not_ready" if exc.status_code == 503 else "invalid_request"
        return error_response(category, str(exc.detail))

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception):
        return error_response("internal", "the service failed to handle the request")

    # --- routes -------------------------------------------------------------
    # Registered before the static mount: Starlette matches in order, so the
    # API answers first and the bundle only catches what is left (FR-013a).

    @app.get(HEALTH_PATH)
    async def health() -> dict:
        """Liveness plus the one condition worth knowing before a demo."""
        return {"status": "ok", "static_root_ready": _static_ready(config.static_root)}

    # The demo's three routes (demo.py). Mounted here rather than in a router
    # module: they read the recorded runs directly, which is what a demo needs
    # and what the session/job endpoints of plan.md would otherwise be built for.
    register_demo_routes(app, config)

    if _static_ready(config.static_root):
        # ponytail: html=True serves index.html for `/` and for directory
        # paths. No cache headers, no custom 404 — the bundle is local files
        # read by one browser on the same machine.
        app.mount("/", StaticFiles(directory=config.static_root, html=True), name="ui")
    else:
        @app.get("/{path:path}")
        async def ui_missing(path: str) -> JSONResponse:
            """Never a built-in UI: with no bundle there is nothing to serve."""
            return _not_ready(config.static_root)

    return app
