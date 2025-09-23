"""Score contracts and sandbox (simplified).

Provides deterministic, timeout-aware execution with minimal schemas
to satisfy router imports and basic functionality.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ContractType(str, Enum):
    DELTA_KEC_V1 = "delta_kec_v1"
    ZUCO_READING_V1 = "zuco_reading_v1"
    EDITORIAL_V1 = "editorial_v1"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ContractInput:
    contract_type: ContractType
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    timeout_seconds: float = 30.0


@dataclass
class ContractOutput:
    execution_id: str
    contract_type: ContractType
    status: ExecutionStatus
    score: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: Optional[str] = None


class _Sandbox:
    def __init__(self) -> None:
        self._history: List[ContractOutput] = []

    def _schema(self, ctype: ContractType) -> Dict[str, Any]:
        return {"type": ctype.value, "schema": {"type": "object", "properties": {}}}

    def get_available_contracts(self) -> List[Dict[str, Any]]:
        return [
            {"type": c.value, "name": c.value, "schema": self._schema(c)["schema"]}
            for c in ContractType
        ]

    def get_contract_schema(self, ctype: ContractType) -> Dict[str, Any]:
        return self._schema(ctype)

    def get_execution_history(
        self, ctype: Optional[ContractType], limit: int
    ) -> List[ContractOutput]:
        items = [
            h for h in self._history if (ctype is None or h.contract_type == ctype)
        ]
        return items[-limit:]

    async def execute_contract(self, cin: ContractInput) -> ContractOutput:
        eid = str(uuid.uuid4())
        start = time.time()
        try:

            async def _run() -> ContractOutput:
                # Deterministic scoring by type
                if cin.contract_type == ContractType.DELTA_KEC_V1:
                    s = float(cin.data.get("target_entropy", 0)) - float(
                        cin.data.get("source_entropy", 0)
                    )
                    score = max(-3.0, min(3.0, s))
                elif cin.contract_type == ContractType.ZUCO_READING_V1:
                    score = 0.5  # placeholder
                else:
                    score = 0.8  # editorial
                return ContractOutput(
                    execution_id=eid,
                    contract_type=cin.contract_type,
                    status=ExecutionStatus.COMPLETED,
                    score=score,
                    confidence=0.75,
                    metadata={"sandbox": True},
                )

            result = await asyncio.wait_for(_run(), timeout=cin.timeout_seconds)
        except asyncio.TimeoutError:
            result = ContractOutput(
                execution_id=eid,
                contract_type=cin.contract_type,
                status=ExecutionStatus.TIMEOUT,
                error_message="Execution timed out",
            )
        except Exception as e:  # pragma: no cover - defensive
            result = ContractOutput(
                execution_id=eid,
                contract_type=cin.contract_type,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
            )

        result.execution_time_ms = (time.time() - start) * 1000.0
        self._history.append(result)
        return result


_SANDBOX = _Sandbox()


def get_sandbox() -> _Sandbox:
    return _SANDBOX
