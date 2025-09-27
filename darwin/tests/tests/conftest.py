import os
import sys
from typing import Generator, Optional

import pytest

# Ensure project root is on sys.path for imports like `services.*`
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Since create_app is in a sibling directory (src), we need to adjust the path
src_path = os.path.join(ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Lazily import FastAPI TestClient and API app to avoid hard dependency for core tests
HAS_API: bool = False
TestClient: Optional[object] = None
create_app = None  # type: ignore[assignment]

try:
    from fastapi.testclient import TestClient as _TestClient  # type: ignore
    from kec_biomat_api.main import create_app as _create_app  # type: ignore

    TestClient = _TestClient
    create_app = _create_app  # type: ignore[assignment]
    HAS_API = True
except Exception:
    HAS_API = False


@pytest.fixture(scope="module")
def client() -> Generator[object, None, None]:
    """
    Create a new FastAPI TestClient that uses the `create_app` factory.
    If API stack is unavailable in the environment, skip API tests gracefully.
    """
    if not HAS_API or TestClient is None or create_app is None:
        pytest.skip("FastAPI stack not installed or API app not available in this environment")
        yield None  # For type consistency; will be skipped above
        return

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

