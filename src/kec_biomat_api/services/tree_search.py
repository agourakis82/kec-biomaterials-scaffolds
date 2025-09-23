"""PUCT-style Tree Search service (simplified).

Implements a minimal PUCT search compatible with the router.
Suitable for unit tests and local runs; extend for full features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, List, Optional, Tuple, TypeVar

S = TypeVar("S")  # State type


class NodeStatus(Enum):
    UNVISITED = "unvisited"
    EXPANDED = "expanded"
    TERMINAL = "terminal"


@dataclass
class TreeSearchConfig:
    max_budget_nodes: int = 300
    max_depth: int = 5
    default_budget: int = 150
    c_puct: float = 1.414
    expansion_threshold: int = 5
    simulation_rollouts: int = 3
    temperature: float = 1.0
    use_progressive_widening: bool = True
    alpha_widening: float = 0.5


@dataclass
class TreeNode(Generic[S]):
    state: S
    parent: Optional[TreeNode[S]] = None
    action_taken: Optional[str] = None
    depth: int = 0
    visits: int = 0
    total_value: float = 0.0
    children: List[TreeNode[S]] = field(default_factory=list)
    status: NodeStatus = NodeStatus.UNVISITED

    @property
    def is_terminal(self) -> bool:
        return self.depth <= 0 or self.status == NodeStatus.TERMINAL

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    @property
    def node_id(self) -> str:
        return f"n{hash((self.state, self.depth)) & 0xFFFF:04x}"

    def puct_score(self, config: TreeSearchConfig) -> float:
        if not self.parent:
            return 0.0
        Q = self.mean_value
        N = self.parent.visits if self.parent.visits > 0 else 1
        n = self.visits
        P = 1.0 / max(1, len(self.parent.children))  # flat prior
        return Q + config.c_puct * P * math.sqrt(N) / (1 + n)


class TestStateEvaluator(Generic[S]):
    """Toy evaluator for string states: expands into plausible next tokens."""

    async def expand(self, state: S) -> List[Tuple[str, S, float]]:
        # Simple branching: next actions are token variants
        base = str(state)
        actions = [f"add:{t}" for t in ["A", "B", "C"]]
        children = [(a, f"{base}->{a}", 1.0 - i * 0.1) for i, a in enumerate(actions)]
        return children

    async def rollout(self, state: S, max_steps: int = 3) -> float:
        # Random but deterministic-ish score by hashing
        rnd = (hash(state) % 1000) / 1000.0
        return rnd


class TreeSearch(Generic[S]):
    def __init__(
        self,
        evaluator: TestStateEvaluator[S],
        config: Optional[TreeSearchConfig] = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or TreeSearchConfig()
        self.root: Optional[TreeNode[S]] = None
        self.nodes_explored = 0

    async def search(
        self, initial_state: S, budget: Optional[int] = None
    ) -> TreeNode[S]:
        budget = budget or self.config.default_budget
        self.root = TreeNode(state=initial_state, depth=self.config.max_depth)
        current_budget = 0

        while current_budget < min(budget, self.config.max_budget_nodes):
            node = await self._select(self.root)
            value = await self._simulate(node)
            self._backpropagate(node, value)
            current_budget += 1
            self.nodes_explored += 1
        return self.root

    async def _select(self, node: TreeNode[S]) -> TreeNode[S]:
        # Expand if needed
        if node.status == NodeStatus.UNVISITED:
            await self._expand(node)
            return node

        # Leaf
        if not node.children or node.depth <= 0:
            return node

        # Choose child with highest PUCT
        best = max(node.children, key=lambda c: c.puct_score(self.config))
        return await self._select(best)

    async def _expand(self, node: TreeNode[S]) -> None:
        node.status = NodeStatus.EXPANDED
        if node.depth <= 0:
            node.status = NodeStatus.TERMINAL
            return
        children = await self.evaluator.expand(node.state)
        for action, new_state, prior in children:
            child = TreeNode(
                state=new_state, parent=node, action_taken=action, depth=node.depth - 1
            )
            node.children.append(child)

    async def _simulate(self, node: TreeNode[S]) -> float:
        # Use rollouts from evaluator
        values = [
            await self.evaluator.rollout(node.state)
            for _ in range(self.config.simulation_rollouts)
        ]
        return sum(values) / max(1, len(values))

    def _backpropagate(self, node: TreeNode[S], value: float) -> None:
        cur: Optional[TreeNode[S]] = node
        while cur is not None:
            cur.visits += 1
            cur.total_value += value
            cur = cur.parent

    def get_best_action_sequence(self, max_length: Optional[int] = None) -> List[str]:
        if not self.root:
            return []
        path: List[str] = []
        node = self.root
        steps = 0
        limit = max_length or self.config.max_depth
        while node.children and steps < limit:
            node = max(node.children, key=lambda c: c.mean_value)
            if node.action_taken:
                path.append(node.action_taken)
            steps += 1
        return path

    def get_search_statistics(self) -> dict:
        def count_nodes(n: TreeNode[S]) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        def depth(n: TreeNode[S]) -> int:
            if not n.children:
                return 0
            return 1 + max(depth(c) for c in n.children)

        if not self.root:
            return {"nodes_count": 0, "max_depth": 0, "search_efficiency": 0.0}
        nodes = count_nodes(self.root)
        d = depth(self.root)
        eff = self.nodes_explored / max(1, nodes)
        return {
            "nodes_count": nodes,
            "max_depth": d,
            "search_efficiency": round(eff, 3),
        }
