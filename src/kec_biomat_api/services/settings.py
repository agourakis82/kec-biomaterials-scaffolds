"""Service-facing settings shim.

Provides a stable `get_settings()` for modules outside `infra.api`.
"""

from typing import Any

try:
    # Prefer the application settings when running as the API package
    from kec_biomat_api.config import settings as _settings
except Exception:
    try:
        # Fallback to module path used in other contexts
        from infra.api.config import settings as _settings
    except Exception:  # pragma: no cover
        _settings = None  # type: ignore


def get_settings() -> Any:
    """Return application settings object.

    This indirection avoids deep import chains in routers/services.
    """
    if _settings is None:
        raise RuntimeError("Settings not initialized")
    return _settings

