"""Spec 005 local web service; module imports are intentionally left to their callers.

Module responsibilities and functional-requirement ranges:

* ``app``: FastAPI app factory; mounts the API routers and the configured static root.
* ``security``: loopback bind, Host/Origin guard, artifact-root allow-list (FR-013c, FR-013d).
* ``session``: backend session store (FR-022a) and run ownership (FR-065).
* ``jobs``: batch job registry and progress (FR-084).
* ``errors``: FR-079 category enum to HTTP status mapping.
* ``schemas``: request/response models for the 17 endpoints.
* ``routes.catalog``: ``/api/catalog/*`` selection enumeration (FR-014-FR-017).
* ``routes.page``: ``/api/page`` page bundle, regions, and provenance.
* ``routes.edit``: ``/api/page/edit`` and ``/api/page/render`` (FR-031-FR-038).
* ``routes.summary``: ``/api/summary/*`` (FR-039-FR-046).
* ``routes.reports``: ``/api/reports/*`` (FR-047-FR-062).
* ``routes.artifacts``: ``/api/artifact``, the only byte-serving route (FR-013d).
* ``routes.jobs``: ``/api/jobs/*`` batch start and status (FR-084).

No submodule is imported here, so package import does not build an app, bind a
socket, or read any configured path.
"""
