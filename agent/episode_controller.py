"""Deterministic guardrails for bounded model-call episodes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from enum import Enum
from typing import Any, Callable, Sequence

from agent.redact import redact_sensitive_text
from agent.task_checkpoint import TaskCheckpoint
from tools.artifact_store import FilesystemArtifactStore
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import (
    PERSISTED_OUTPUT_TAG,
    maybe_persist_tool_result,
)


_ARTIFACT_REFERENCE_RE = re.compile(
    r"Artifact ID:\s*(artifact_[0-9a-f]{20}).*?"
    r"SHA-256:\s*([0-9a-f]{64})",
    re.DOTALL,
)


class EpisodeAction(str, Enum):
    CONTINUE = "continue"
    CHECKPOINT = "checkpoint"
    ROLLOVER = "rollover"
    STOP = "stop"


class EpisodeController:
    """Track episode and task budgets independently of transcript state."""

    def __init__(
        self,
        *,
        checkpoint_interval: int = 20,
        rollover_interval: int = 40,
        max_episodes: int = 5,
        max_total_iterations: int = 200,
        max_wall_seconds: float = 7_200,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if rollover_interval <= checkpoint_interval:
            raise ValueError("rollover_interval must exceed checkpoint_interval")
        if max_episodes <= 0 or max_total_iterations <= 0 or max_wall_seconds <= 0:
            raise ValueError("episode task limits must be positive")
        self.checkpoint_interval = checkpoint_interval
        self.rollover_interval = rollover_interval
        self.max_episodes = max_episodes
        self.max_total_iterations = max_total_iterations
        self.max_wall_seconds = max_wall_seconds
        self._clock = clock
        self._started_at = clock()
        self.episode_number = 1
        self.episode_iterations = 0
        self.total_iterations = 0

    @property
    def wall_clock_elapsed(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def record_iteration(self) -> EpisodeAction:
        self.episode_iterations += 1
        self.total_iterations += 1
        if (
            self.total_iterations >= self.max_total_iterations
            or self.wall_clock_elapsed >= self.max_wall_seconds
        ):
            return EpisodeAction.STOP
        if self.episode_iterations >= self.rollover_interval:
            if self.episode_number >= self.max_episodes:
                return EpisodeAction.STOP
            return EpisodeAction.ROLLOVER
        if self.episode_iterations % self.checkpoint_interval == 0:
            return EpisodeAction.CHECKPOINT
        return EpisodeAction.CONTINUE

    def begin_next_episode(self) -> None:
        if self.episode_number >= self.max_episodes:
            raise RuntimeError("maximum episode count reached")
        self.episode_number += 1
        self.episode_iterations = 0


def episode_controller_from_config(
    config: dict[str, Any] | None,
    *,
    session_id: str | None = None,
) -> EpisodeController | None:
    """Construct an opt-in controller from the merged profile config."""
    config = config or {}
    if not config.get("enabled", False):
        return None
    session_allowlist = config.get("session_allowlist")
    if session_allowlist is not None:
        if not isinstance(session_allowlist, list) or any(
            not isinstance(value, str) or not value
            for value in session_allowlist
        ):
            raise ValueError("session_allowlist must be null or a list of non-empty strings")
        if session_id is None or session_id not in session_allowlist:
            return None
    return EpisodeController(
        checkpoint_interval=int(config.get("checkpoint_interval", 20)),
        rollover_interval=int(config.get("rollover_interval", 40)),
        max_episodes=int(config.get("max_episodes", 5)),
        max_total_iterations=int(config.get("max_total_iterations", 200)),
        max_wall_seconds=float(config.get("max_wall_seconds", 7_200)),
    )


def build_mechanical_checkpoint(
    *,
    agent: Any,
    controller: EpisodeController,
    objective: str,
    messages: Sequence[dict[str, Any]],
    task_id: str | None = None,
) -> TaskCheckpoint:
    """Build bounded continuation state from deterministic runtime sources."""
    todos: Sequence[dict[str, Any]] = []
    todo_store = getattr(agent, "_todo_store", None)
    if todo_store is not None:
        try:
            todos = todo_store.read() or []
        except Exception:
            todos = []

    def _safe_text(value: Any, limit: int = 512) -> str:
        text = str(value or "")[:limit]
        return redact_sensitive_text(text, force=True)

    completed = [
        _safe_text(item.get("content"))
        for item in todos[:32]
        if item.get("status") == "completed"
    ]
    failed = [
        _safe_text(item.get("content"))
        for item in todos[:32]
        if item.get("status") == "cancelled"
    ]
    next_items = [
        _safe_text(item.get("content"))
        for item in todos[:32]
        if item.get("status") in {"in_progress", "pending"}
    ]

    # Carry forward the prior checkpoint's bounded mechanical ledger, then add
    # tool results from this active episode. Arguments and result content are
    # deliberately excluded. A digest of the provider-generated call ID gives
    # stable deduplication without carrying raw inputs across rollover.
    prior_completed: list[str] = []
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    checkpoint_loader = getattr(session_db, "load_latest_task_checkpoint", None)
    if task_id and callable(checkpoint_loader) and session_id:
        prior = checkpoint_loader(session_id, task_id)
        if prior is not None:
            stored_steps = prior.get("completed_steps", [])
            if not isinstance(stored_steps, list) or any(
                not isinstance(step, str) for step in stored_steps
            ):
                raise ValueError("stored checkpoint contains invalid completed_steps")
            prior_completed = stored_steps[:64]

    tool_steps: list[str] = []
    for message in messages[-64:]:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        call_digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:12]
        tool_name = _safe_text(message.get("tool_name") or "unknown", 128)
        tool_steps.append(f"[tool] {call_digest} {tool_name} result recorded")

    cumulative_completed: list[str] = []
    for step in [*prior_completed, *completed, *tool_steps]:
        if step and step not in cumulative_completed:
            cumulative_completed.append(step)
    cumulative_completed = cumulative_completed[-64:]

    verified_artifacts: list[dict[str, str]] = []
    seen_artifacts: set[str] = set()
    store = FilesystemArtifactStore()
    for message in messages[-64:]:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or PERSISTED_OUTPUT_TAG not in content:
            continue
        # Durable artifact references are compact; cap scanning so checkpoint
        # creation cannot become a second unbounded raw-output path. A pointer
        # is carried forward only after resolving it in the canonical store.
        for artifact_id, digest in _ARTIFACT_REFERENCE_RE.findall(
            content[:32_768]
        ):
            if artifact_id in seen_artifacts:
                continue
            try:
                stored = store.resolve(artifact_id)
            except Exception:
                continue
            if stored.sha256 != digest:
                continue
            seen_artifacts.add(artifact_id)
            verified_artifacts.append(
                {"artifact_id": artifact_id, "sha256": digest}
            )
            if len(verified_artifacts) >= 32:
                break

    continuation_notes: list[str] = []
    for message in reversed(messages[-16:]):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            continuation_notes.append(_safe_text(content, 2_000))
            break

    continuation_prompts: list[str] = []
    for message in reversed(messages[-16:]):
        if message.get("role") != "user":
            continue
        if not (
            message.get("_empty_recovery_synthetic")
            or message.get("_verification_stop_synthetic")
            or message.get("_pre_verify_synthetic")
            or str(message.get("content") or "").startswith("[System: Continue now")
        ):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            continuation_prompts.append(_safe_text(content, 2_000))
            break

    return TaskCheckpoint(
        objective=_safe_text(objective, 4_000),
        success_criteria=[],
        constraints=[],
        completed_steps=cumulative_completed,
        verified_artifacts=verified_artifacts,
        decisions=continuation_notes,
        failed_approaches=failed,
        active_processes=[],
        delegations=[],
        open_questions=continuation_prompts,
        next_action=(next_items[0] if next_items else "Continue toward the objective"),
        git_state={},
        episode_number=controller.episode_number,
        episode_iterations=controller.episode_iterations,
        total_iterations=controller.total_iterations,
        wall_clock_elapsed=controller.wall_clock_elapsed,
    )


def process_episode_iteration(
    *,
    agent: Any,
    controller: EpisodeController,
    objective: str,
    task_id: str,
    messages: list[dict[str, Any]],
) -> tuple[EpisodeAction, list[dict[str, Any]]]:
    """Record a completed model/tool iteration and apply its boundary action."""
    action = controller.record_iteration()
    return action, apply_episode_boundary(
        agent=agent,
        controller=controller,
        action=action,
        objective=objective,
        task_id=task_id,
        messages=messages,
    )


def _canonical_artifact_pointer(artifact_id: str, sha256: str) -> str:
    return (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"Artifact ID: {artifact_id}\n"
        f"SHA-256: {sha256}\n"
        "Use artifact_get with this artifact ID, offset, and limit to retrieve "
        "verified bounded sections.\n"
        "</persisted-output>"
    )


def _prepare_pending_tool_tail(
    agent: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Artifactize the unconsumed final tool results for a bounded rollover."""
    prepared = copy.deepcopy(messages)
    if not prepared or prepared[-1].get("role") != "tool":
        # Empty-response/model-retry paths have no unconsumed tool output.
        return prepared, []

    assistant_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            assistant_index = index
            break
    if assistant_index is None:
        raise RuntimeError("rollover requires a complete pending tool-call group")

    store = FilesystemArtifactStore()
    tail = prepared[assistant_index:]
    if any(message.get("role") != "tool" for message in tail[1:]):
        raise RuntimeError("rollover pending tail contains an invalid role sequence")
    tool_results = tail[1:]
    if not tool_results:
        raise RuntimeError("rollover requires pending tool results")
    if len(tool_results) > 32:
        raise RuntimeError("rollover pending tool group exceeds 32 results")

    assistant_calls = tail[0].get("tool_calls") or []
    if len(assistant_calls) != len(tool_results):
        raise RuntimeError("rollover tool-call/result group is incomplete")
    sanitized_calls: list[dict[str, Any]] = []
    for tool_call in assistant_calls:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            raise RuntimeError("rollover contains an invalid tool call")
        call_id = str(tool_call.get("id") or "")
        function_name = str(function.get("name") or "")
        if not call_id or len(call_id) > 256 or not function_name or len(function_name) > 256:
            raise RuntimeError("rollover contains an invalid tool call identity")
        sanitized_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": function_name, "arguments": "{}"},
            }
        )
    assistant_content = redact_sensitive_text(
        str(tail[0].get("content") or "")[:4_000], force=True
    )
    tail[0] = {
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": sanitized_calls,
    }

    for index, message in enumerate(tool_results):
        expected_call_id = sanitized_calls[index]["id"]
        if str(message.get("tool_call_id") or "") != expected_call_id:
            raise RuntimeError("rollover tool result does not match its tool call")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("rollover cannot preserve non-text tool results")
        if not content:
            continue
        if PERSISTED_OUTPUT_TAG in content:
            artifact_match = _ARTIFACT_REFERENCE_RE.search(content[:32_768])
            if artifact_match is not None:
                artifact_id, advertised_digest = artifact_match.groups()
                try:
                    stored = store.resolve(artifact_id)
                except Exception as exc:
                    raise RuntimeError(
                        "pending artifact reference could not be verified"
                    ) from exc
                if stored.sha256 != advertised_digest:
                    raise RuntimeError("pending artifact reference digest mismatch")
                message["content"] = _canonical_artifact_pointer(
                    artifact_id, advertised_digest
                )
                continue
        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=str(message.get("tool_name") or "episode_rollover"),
            tool_use_id=str(message.get("tool_call_id") or f"rollover_{index}"),
            env=getattr(agent, "env", None),
            config=BudgetConfig(preview_size=0),
            threshold=0,
            session_id=getattr(agent, "session_id", None),
            artifact_store=store,
        )
        artifact_match = _ARTIFACT_REFERENCE_RE.search(replacement[:32_768])
        if artifact_match is None:
            raise RuntimeError("pending tool result could not be durably artifactized")
        message["content"] = _canonical_artifact_pointer(*artifact_match.groups())

    for index, message in enumerate(tool_results, start=1):
        tail[index] = {
            "role": "tool",
            "tool_call_id": str(message.get("tool_call_id")),
            "tool_name": str(message.get("tool_name") or "")[:256],
            "content": message.get("content", ""),
        }
    prepared[assistant_index:] = tail
    return prepared, tail


def apply_episode_boundary(
    *,
    agent: Any,
    controller: EpisodeController,
    action: EpisodeAction,
    objective: str,
    task_id: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Persist a mechanical checkpoint and optionally start a fresh context."""
    if action not in {
        EpisodeAction.CHECKPOINT,
        EpisodeAction.ROLLOVER,
        EpisodeAction.STOP,
    }:
        return messages
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db is None or not session_id:
        raise RuntimeError("bounded episode checkpoint requires a session database")

    checkpoint_messages = messages
    pending_tail: list[dict[str, Any]] = []
    if action is EpisodeAction.ROLLOVER:
        checkpoint_messages, pending_tail = _prepare_pending_tool_tail(agent, messages)

    checkpoint = build_mechanical_checkpoint(
        agent=agent,
        controller=controller,
        objective=objective,
        messages=checkpoint_messages,
        task_id=task_id,
    )
    if action in {EpisodeAction.ROLLOVER, EpisodeAction.STOP}:
        agent._flush_messages_to_session_db(messages)
    session_db.save_task_checkpoint(session_id, task_id, checkpoint.to_dict())

    if action in {EpisodeAction.CHECKPOINT, EpisodeAction.STOP}:
        return messages

    controller.begin_next_episode()
    continuation = (
        "[BOUNDED EPISODE CONTINUATION]\n"
        "The prior episode is archived. Continue automatically from this "
        "structured checkpoint. Do not repeat completed work. Retrieve only "
        "the bounded artifact sections needed for the next action.\n"
        "<task-checkpoint>\n"
        f"{json.dumps(checkpoint.to_dict(), ensure_ascii=False, sort_keys=True)}\n"
        "</task-checkpoint>"
    )
    agent._emit_status(
        "Task checkpointed after "
        f"{controller.total_iterations} iterations. Continuing in bounded "
        f"episode {controller.episode_number}/{controller.max_episodes}."
    )
    return [{"role": "user", "content": continuation}, *pending_tail]


__all__ = [
    "EpisodeAction",
    "EpisodeController",
    "apply_episode_boundary",
    "build_mechanical_checkpoint",
    "episode_controller_from_config",
    "process_episode_iteration",
]
