import os
import sys
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path for imports like `services.*`
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Since create_app is in a sibling directory (src), we need to adjust the path
sys.path.insert(0, os.path.join(ROOT, "src"))

from kec_biomat_api.main import create_app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """
    Create a new FastAPI TestClient that uses the `create_app` factory.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

