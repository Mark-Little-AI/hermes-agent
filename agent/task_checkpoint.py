"""Structured continuation state for bounded execution episodes."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


MAX_CHECKPOINT_BYTES = 65_536
CHECKPOINT_FIELDS = frozenset(
    {
        "objective",
        "success_criteria",
        "constraints",
        "completed_steps",
        "verified_artifacts",
        "decisions",
        "failed_approaches",
        "active_processes",
        "delegations",
        "open_questions",
        "next_action",
        "git_state",
        "episode_number",
        "episode_iterations",
        "total_iterations",
        "wall_clock_elapsed",
        "schema_version",
    }
)
_STRING_LIST_FIELDS = {
    "success_criteria",
    "constraints",
    "completed_steps",
    "decisions",
    "failed_approaches",
    "open_questions",
}
_ARTIFACT_ID_RE = re.compile(r"artifact_[0-9a-f]{20}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _validate_string_list(name: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError(f"checkpoint field {name} must be a list of at most 64 strings")
    if any(not isinstance(item, str) or len(item) > 2_000 for item in value):
        raise ValueError(f"checkpoint field {name} contains an invalid item")


def _validate_scalar_mappings(name: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError(f"checkpoint field {name} must be a bounded list")
    for item in value:
        if not isinstance(item, dict) or len(item) > 16:
            raise ValueError(f"checkpoint field {name} contains an invalid mapping")
        for key, scalar in item.items():
            if not isinstance(key, str) or len(key) > 64:
                raise ValueError(f"checkpoint field {name} contains an invalid key")
            if not isinstance(scalar, (str, int, float, bool, type(None))):
                raise ValueError(f"checkpoint field {name} contains nested state")
            if isinstance(scalar, str) and len(scalar) > 2_000:
                raise ValueError(f"checkpoint field {name} contains an oversized value")


def validate_checkpoint_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Reject transcript-shaped, unknown, or unbounded checkpoint state."""
    unknown = set(checkpoint) - CHECKPOINT_FIELDS
    missing = CHECKPOINT_FIELDS - set(checkpoint)
    if unknown:
        raise ValueError(f"unsupported checkpoint fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing checkpoint fields: {', '.join(sorted(missing))}")
    payload = dict(checkpoint)

    if not isinstance(payload["objective"], str) or len(payload["objective"]) > 4_000:
        raise ValueError("checkpoint objective must be a bounded string")
    if not isinstance(payload["next_action"], str) or len(payload["next_action"]) > 2_000:
        raise ValueError("checkpoint next_action must be a bounded string")
    for name in _STRING_LIST_FIELDS:
        _validate_string_list(name, payload[name])

    artifacts = payload["verified_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) > 64:
        raise ValueError("verified_artifacts must be a bounded list")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"artifact_id", "sha256"}:
            raise ValueError("verified_artifacts contains an invalid reference")
        if not _ARTIFACT_ID_RE.fullmatch(artifact.get("artifact_id", "")):
            raise ValueError("verified_artifacts contains an invalid artifact ID")
        if not _SHA256_RE.fullmatch(artifact.get("sha256", "")):
            raise ValueError("verified_artifacts contains an invalid digest")

    _validate_scalar_mappings("active_processes", payload["active_processes"])
    _validate_scalar_mappings("delegations", payload["delegations"])
    git_state = payload["git_state"]
    if not isinstance(git_state, dict) or len(git_state) > 16:
        raise ValueError("git_state must be a bounded mapping")
    _validate_scalar_mappings("git_state", [git_state])

    for name in ("episode_number", "episode_iterations", "total_iterations"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint field {name} must be a non-negative integer")
    if payload["episode_number"] < 1:
        raise ValueError("episode_number must be at least one")
    elapsed = payload["wall_clock_elapsed"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
        or not math.isfinite(float(elapsed))
    ):
        raise ValueError("wall_clock_elapsed must be finite and non-negative")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("unsupported checkpoint schema_version")

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint payload is not JSON serializable") from exc
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"checkpoint exceeds {MAX_CHECKPOINT_BYTES} byte limit"
        )
    return payload


@dataclass(frozen=True)
class TaskCheckpoint:
    """Compact task state that can resume work without transcript replay."""

    objective: str
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    verified_artifacts: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)
    active_processes: list[dict[str, Any]] = field(default_factory=list)
    delegations: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_action: str = ""
    git_state: dict[str, Any] = field(default_factory=dict)
    episode_number: int = 1
    episode_iterations: int = 0
    total_iterations: int = 0
    wall_clock_elapsed: float = 0.0
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["TaskCheckpoint"]
