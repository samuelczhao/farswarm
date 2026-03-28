"""Twitter-like platform wrapping OASIS."""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path
from typing import Protocol

from farswarm.core.types import AgentProfile
from farswarm.simulation.platforms.base import SimulationPlatform

logger = logging.getLogger(__name__)

OASIS_CSV_HEADERS = ["user_id", "name", "username", "user_char", "description"]

try:
    import oasis  # type: ignore[import-untyped]
    OASIS_AVAILABLE = True
except ImportError:
    OASIS_AVAILABLE = False


class OasisEnv(Protocol):
    """Protocol for OASIS environment."""

    async def step(self, actions: dict[object, object]) -> object: ...


class TwitterPlatform(SimulationPlatform):
    """Twitter-like platform backed by OASIS or fallback."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._env: OasisEnv | None = None
        self._agents: list[AgentProfile] = []
        self._oasis_agents: list[object] = []
        self._posts: list[str] = []
        self._db_path = db_path

    async def setup(self, agents: list[AgentProfile]) -> None:
        """Initialize platform and write agent CSV for OASIS."""
        self._agents = agents
        if OASIS_AVAILABLE:
            await self._setup_oasis(agents)
        else:
            logger.warning("OASIS not installed; using fallback simulation")

    async def step(self, active_agents: list[AgentProfile]) -> list[dict[str, object]]:
        """Execute one step with active agents."""
        if OASIS_AVAILABLE and self._env is not None:
            return await self._step_oasis(active_agents)
        return self._step_fallback(active_agents)

    def get_trending_content(self) -> list[str]:
        """Return recent posts as trending content."""
        return self._posts[-50:] if self._posts else []

    async def _setup_oasis(self, agents: list[AgentProfile]) -> None:
        """Configure OASIS environment."""
        csv_path = self._write_agent_csv(agents)
        agent_graph = oasis.generate_twitter_agent_graph(csv_path)  # type: ignore[attr-defined]
        self._env = oasis.make("twitter", agent_graph=agent_graph)  # type: ignore[attr-defined]
        self._oasis_agents = list(agent_graph.agents)

    async def _step_oasis(self, active_agents: list[AgentProfile]) -> list[dict[str, object]]:
        """Run one OASIS step."""
        LLMAction = oasis.LLMAction  # type: ignore[attr-defined] # noqa: N806
        active_ids = {a.agent_id for a in active_agents}
        actions = {}
        for i, oasis_agent in enumerate(self._oasis_agents):
            if i in active_ids:
                actions[oasis_agent] = LLMAction()
        result = await self._env.step(actions)  # type: ignore[union-attr]
        return self._parse_oasis_result(result)

    def _step_fallback(self, active_agents: list[AgentProfile]) -> list[dict[str, object]]:
        """Generate synthetic actions without OASIS."""
        actions: list[dict[str, object]] = []
        for agent in active_agents:
            post = f"[{agent.username}] Synthetic post about {agent.archetype.label}"
            self._posts.append(post)
            actions.append({
                "agent_id": agent.agent_id,
                "action": "create_post",
                "content": post,
            })
        return actions

    def _write_agent_csv(self, agents: list[AgentProfile]) -> Path:
        """Write OASIS-format agent CSV."""
        csv_path = Path(tempfile.mktemp(suffix=".csv"))
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(OASIS_CSV_HEADERS)
            for agent in agents:
                user_char = f"{agent.archetype.description}\n{agent.persona}"
                writer.writerow([
                    agent.agent_id, agent.name, agent.username,
                    user_char, agent.bio,
                ])
        return csv_path

    def _parse_oasis_result(self, result: object) -> list[dict[str, object]]:
        """Convert OASIS step result to action dicts."""
        if isinstance(result, list):
            return [{"raw": r} for r in result]
        return [{"raw": result}]
