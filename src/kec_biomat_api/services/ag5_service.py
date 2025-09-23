"""Minimal AG5 data service stub."""

from typing import Any, Dict, List


class _AG5Service:
    async def list_datasets(self) -> List[str]:
        return ["ag5_sample"]

    async def get_dataset_info(self, name: str) -> Dict[str, Any]:
        return {"name": name, "records": 0}

    async def search_datasets(self, q: str) -> List[Dict[str, Any]]:
        return []

    async def health_check(self) -> Dict[str, str]:
        return {"status": "ready"}


ag5_service = _AG5Service()
