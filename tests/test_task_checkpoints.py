import sqlite3

import pytest

from agent.task_checkpoint import TaskCheckpoint
from hermes_state import SessionDB


def _checkpoint(*, next_action: str, total_iterations: int = 20) -> TaskCheckpoint:
    return TaskCheckpoint(
        objective="Implement bounded execution episodes",
        success_criteria=["Fresh episode resumes from verified state"],
        constraints=["Do not modify the live runtime"],
        completed_steps=["Durable artifacts committed"],
        verified_artifacts=[{"artifact_id": "artifact_0123456789abcdefabcd", "sha256": "a" * 64}],
        decisions=["Checkpoint mechanically before rollover"],
        failed_approaches=["Raw transcript replay"],
        active_processes=[],
        delegations=[],
        open_questions=[],
        next_action=next_action,
        git_state={"branch": "feat/bounded-context-artifacts", "clean": True},
        episode_number=1,
        episode_iterations=total_iterations,
        total_iterations=total_iterations,
        wall_clock_elapsed=123.5,
    )


def test_latest_structured_checkpoint_round_trips_by_session_and_task(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", "test")
    first = _checkpoint(next_action="write rollover test")
    second = _checkpoint(next_action="implement rollover", total_iterations=21)

    first_id = db.save_task_checkpoint("session-1", "task-1", first.to_dict())
    second_id = db.save_task_checkpoint("session-1", "task-1", second.to_dict())

    assert second_id > first_id
    assert db.load_latest_task_checkpoint("session-1", "task-1") == second.to_dict()
    assert db.load_latest_task_checkpoint("session-1", "other-task") is None
    db.close()


def test_checkpoint_storage_rejects_transcript_and_oversized_payloads(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", "test")
    valid = _checkpoint(next_action="continue").to_dict()

    with pytest.raises(ValueError, match="unsupported checkpoint fields"):
        db.save_task_checkpoint(
            "session-1",
            "task-1",
            {**valid, "raw_transcript": "must not be persisted"},
        )

    with pytest.raises(ValueError, match="bounded string|checkpoint exceeds"):
        db.save_task_checkpoint(
            "session-1",
            "task-1",
            {**valid, "objective": "x" * 70_000},
        )

    assert db.load_latest_task_checkpoint("session-1", "task-1") is None
    db.close()


def test_checkpoint_storage_rejects_invalid_field_types_and_artifacts(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", "test")
    valid = _checkpoint(next_action="continue").to_dict()

    invalid_payloads = [
        {**valid, "objective": ["not", "a", "string"]},
        {**valid, "completed_steps": [{"raw": "transcript-shaped"}]},
        {
            **valid,
            "verified_artifacts": [
                {"artifact_id": "../../escape", "sha256": "not-a-digest"}
            ],
        },
        {**valid, "total_iterations": -1},
        {**valid, "schema_version": True},
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            db.save_task_checkpoint("session-1", "task-1", payload)

    assert db.load_latest_task_checkpoint("session-1", "task-1") is None
    db.close()


def test_existing_database_without_checkpoint_table_is_migrated(tmp_path):
    db_path = tmp_path / "state.db"
    original = SessionDB(db_path=db_path)
    original.create_session("session-1", "test")
    original.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE task_checkpoints")

    reopened = SessionDB(db_path=db_path)
    checkpoint_id = reopened.save_task_checkpoint(
        "session-1", "task-1", _checkpoint(next_action="resume").to_dict()
    )

    assert checkpoint_id > 0
    latest = reopened.load_latest_task_checkpoint("session-1", "task-1")
    assert latest is not None
    assert latest["next_action"] == "resume"
    reopened.close()
