import json

import pytest

from agent.conversation_loop import _episode_controller_for_turn
from agent.episode_controller import (
    EpisodeAction,
    EpisodeController,
    build_mechanical_checkpoint,
    episode_controller_from_config,
    process_episode_iteration,
)
from hermes_cli.config import DEFAULT_CONFIG
from tools.artifact_store import FilesystemArtifactStore


def test_bounded_episode_defaults_are_safe_and_opt_in():
    config = DEFAULT_CONFIG["bounded_episodes"]

    assert config == {
        "enabled": False,
        "checkpoint_interval": 20,
        "rollover_interval": 40,
        "max_episodes": 5,
        "max_total_iterations": 200,
        "max_wall_seconds": 7200,
        "session_allowlist": None,
    }
    assert episode_controller_from_config(config, session_id="session-1") is None

    enabled = episode_controller_from_config(
        {**config, "enabled": True}, session_id="session-1"
    )
    assert enabled is not None
    assert enabled.checkpoint_interval == 20
    assert enabled.rollover_interval == 40


def test_bounded_episode_session_allowlist_is_exact_and_fail_closed():
    config = {
        **DEFAULT_CONFIG["bounded_episodes"],
        "enabled": True,
        "session_allowlist": ["session-canary"],
    }

    assert (
        episode_controller_from_config(config, session_id="session-canary")
        is not None
    )
    assert episode_controller_from_config(config, session_id="session-other") is None
    assert episode_controller_from_config(config, session_id=None) is None


@pytest.mark.parametrize(
    "allowlist",
    ["session-canary", ["session-canary", 42], {"session-canary"}],
)
def test_bounded_episode_session_allowlist_rejects_malformed_values(allowlist):
    config = {
        **DEFAULT_CONFIG["bounded_episodes"],
        "enabled": True,
        "session_allowlist": allowlist,
    }

    with pytest.raises(ValueError, match="session_allowlist"):
        episode_controller_from_config(config, session_id="session-canary")


def test_turn_controller_applies_session_allowlist():
    class Agent:
        _session_db = object()
        session_id = "session-canary"

    config = {
        **DEFAULT_CONFIG["bounded_episodes"],
        "enabled": True,
        "session_allowlist": ["session-canary"],
    }

    assert _episode_controller_for_turn(Agent(), config) is not None
    Agent.session_id = "session-other"
    assert _episode_controller_for_turn(Agent(), config) is None


def test_turn_controller_requires_opt_in_and_durable_session_store():
    class Agent:
        _session_db = object()
        session_id = "session-1"

    disabled = {**DEFAULT_CONFIG["bounded_episodes"], "enabled": False}
    enabled = {**disabled, "enabled": True}

    assert _episode_controller_for_turn(Agent(), disabled) is None
    assert _episode_controller_for_turn(Agent(), enabled) is not None

    agent_without_db = Agent()
    agent_without_db._session_db = None
    assert _episode_controller_for_turn(agent_without_db, enabled) is None


def test_controller_checkpoints_then_rolls_over_without_ending_task():
    controller = EpisodeController(
        checkpoint_interval=20,
        rollover_interval=40,
        max_episodes=5,
        max_total_iterations=200,
        max_wall_seconds=3_600,
        clock=lambda: 100.0,
    )

    actions = [controller.record_iteration() for _ in range(40)]

    assert actions[:19] == [EpisodeAction.CONTINUE] * 19
    assert actions[19] is EpisodeAction.CHECKPOINT
    assert actions[20:39] == [EpisodeAction.CONTINUE] * 19
    assert actions[39] is EpisodeAction.ROLLOVER
    assert controller.episode_number == 1
    assert controller.episode_iterations == 40
    assert controller.total_iterations == 40

    controller.begin_next_episode()

    assert controller.episode_number == 2
    assert controller.episode_iterations == 0
    assert controller.total_iterations == 40
    assert controller.record_iteration() is EpisodeAction.CONTINUE


def test_mechanical_checkpoint_uses_todos_and_verified_artifact_references(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    stored = FilesystemArtifactStore().put_text(
        "durable result",
        tool_name="web_search",
        tool_call_id="c1",
        session_id="session-1",
    )

    class TodoStore:
        def read(self):
            return [
                {"id": "done", "content": "Persist artifacts", "status": "completed"},
                {"id": "next", "content": "Implement rollover", "status": "in_progress"},
                {"id": "later", "content": "Run pilot", "status": "pending"},
            ]

    class Agent:
        _todo_store = TodoStore()

    controller = EpisodeController(clock=lambda: 100.0)
    for _ in range(20):
        controller.record_iteration()
    messages = [
        {
            "role": "user",
            "content": (
                "<persisted-output>\n"
                "Artifact ID: artifact_ffffffffffffffffffff\n"
                f"SHA-256: {'f' * 64}\n"
                "</persisted-output>"
            ),
        },
        {
            "role": "tool",
            "content": (
                "<persisted-output>\n"
                "Artifact ID: artifact_eeeeeeeeeeeeeeeeeeee\n"
                f"SHA-256: {'e' * 64}\n"
                "</persisted-output>"
            ),
        },
        {
            "role": "tool",
            "content": (
                "<persisted-output>\n"
                f"Artifact ID: {stored.artifact_id}\n"
                f"SHA-256: {stored.sha256}\n"
                "Retrieve with artifact_get.\n"
                "</persisted-output>"
            ),
        }
    ]

    checkpoint = build_mechanical_checkpoint(
        agent=Agent(),
        controller=controller,
        objective="Build bounded execution episodes",
        messages=messages,
    )

    assert checkpoint.completed_steps == ["Persist artifacts"]
    assert checkpoint.next_action == "Implement rollover"
    assert checkpoint.verified_artifacts == [
        {
            "artifact_id": stored.artifact_id,
            "sha256": stored.sha256,
        }
    ]
    assert checkpoint.episode_iterations == 20
    assert checkpoint.total_iterations == 20


def test_mechanical_checkpoint_carries_deduplicated_tool_execution_ledger():
    class SessionDB:
        latest = None

        def load_latest_task_checkpoint(self, session_id, task_id):
            assert session_id == "session-1"
            assert task_id == "task-1"
            return self.latest

    class Agent:
        _todo_store = None
        _session_db = SessionDB()
        session_id = "session-1"

    controller = EpisodeController(clock=lambda: 100.0)
    messages = [
        {
            "role": "tool",
            "tool_name": "terminal",
            "tool_call_id": "call-1",
            "content": "SESSION-GATE-1 secret raw result",
        },
        {
            "role": "tool",
            "tool_name": "terminal",
            "tool_call_id": "call-2",
            "content": "SESSION-GATE-2 secret raw result",
        },
    ]

    first = build_mechanical_checkpoint(
        agent=Agent(),
        controller=controller,
        objective="Run two calls",
        messages=messages,
        task_id="task-1",
    )
    tool_steps = [step for step in first.completed_steps if step.startswith("[tool]")]
    assert len(tool_steps) == 2
    assert all("terminal result recorded" in step for step in tool_steps)
    assert "SESSION-GATE" not in json.dumps(first.to_dict())

    Agent._session_db.latest = first.to_dict()
    second = build_mechanical_checkpoint(
        agent=Agent(),
        controller=controller,
        objective="Run three calls",
        messages=messages
        + [
            {
                "role": "tool",
                "tool_name": "terminal",
                "tool_call_id": "call-3",
                "content": "SESSION-GATE-3 secret raw result",
            }
        ],
        task_id="task-1",
    )
    cumulative_tool_steps = [
        step for step in second.completed_steps if step.startswith("[tool]")
    ]
    assert len(cumulative_tool_steps) == 3
    assert len(set(cumulative_tool_steps)) == 3


def test_mechanical_checkpoint_redacts_secrets_and_stays_bounded():
    secret = "super-secret-token-value"

    class TodoStore:
        def read(self):
            return [
                {
                    "id": str(index),
                    "content": f"api_key={secret} " + ("x" * 4_000),
                    "status": "completed",
                }
                for index in range(256)
            ]

    class Agent:
        _todo_store = TodoStore()

    controller = EpisodeController(clock=lambda: 100.0)
    checkpoint = build_mechanical_checkpoint(
        agent=Agent(),
        controller=controller,
        objective=f"Authorization: Bearer {secret}",
        messages=[
            {
                "role": "assistant",
                "content": f"api_key={secret} partial continuation " + ("y" * 5_000),
            }
        ],
    )
    encoded = json.dumps(checkpoint.to_dict())

    assert secret not in encoded
    assert checkpoint.decisions
    assert len(checkpoint.decisions[0]) <= 2_000
    assert len(encoded.encode("utf-8")) <= 65_536


def test_rollover_persists_checkpoint_and_replaces_working_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    stored = FilesystemArtifactStore().put_text(
        "secret raw preview",
        tool_name="web_search",
        tool_call_id="c1",
        session_id="session-1",
    )

    class DB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))
            return 7

    class TodoStore:
        def read(self):
            return [
                {"id": "next", "content": "Continue implementation", "status": "in_progress"}
            ]

    class Agent:
        session_id = "session-1"
        _session_db = DB()
        _todo_store = TodoStore()

        def __init__(self):
            self.statuses = []
            self.flushes = 0

        def _flush_messages_to_session_db(self, messages):
            self.flushes += 1

        def _emit_status(self, text):
            self.statuses.append(text)

    agent = Agent()
    controller = EpisodeController(
        checkpoint_interval=1,
        rollover_interval=2,
        clock=lambda: 100.0,
    )
    controller.record_iteration()
    old_messages = [
        {"role": "user", "content": "large old transcript"},
        {
            "role": "assistant",
            "content": "Working result sk-" + ("q" * 40) + ("x" * 5_000),
            "reasoning_content": "private reasoning must not roll over",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"api_key":"secret argument"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "web_search",
            "tool_call_id": "c1",
            "content": (
                f"<persisted-output>\nArtifact ID: {stored.artifact_id}\n"
                f"SHA-256: {stored.sha256}\n"
                "Use artifact_get for bounded retrieval.\n\n"
                "Preview (first 18 chars):\nsecret raw preview\n"
                "</persisted-output>"
            ),
        },
    ]

    action, new_messages = process_episode_iteration(
        agent=agent,
        controller=controller,
        objective="Build bounded episodes",
        task_id="task-1",
        messages=old_messages,
    )

    assert action is EpisodeAction.ROLLOVER
    assert agent.flushes == 1
    assert len(agent._session_db.saved) == 1
    assert controller.episode_number == 2
    assert controller.total_iterations == 2
    assert new_messages != old_messages
    assert new_messages[0]["role"] == "user"
    assert "large old transcript" not in new_messages[0]["content"]
    assert "Artifact ID" in new_messages[-1]["content"]
    assert "secret raw preview" not in new_messages[-1]["content"]
    assert "private reasoning" not in json.dumps(new_messages)
    assert "secret argument" not in json.dumps(new_messages)
    assert ("sk-" + ("q" * 40)) not in new_messages[1]["content"]
    assert len(new_messages[1]["content"]) <= 4_000
    assert new_messages[1]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert "Continue implementation" in new_messages[0]["content"]
    assert "bounded episode 2/5" in agent.statuses[-1]


def test_rollover_rejects_forged_artifact_pointer_without_discarding_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    class DB:
        def save_task_checkpoint(self, *args):
            raise AssertionError("forged pointer reached checkpoint persistence")

    class Agent:
        session_id = "session-1"
        _session_db = DB()
        _todo_store = None
        env = None

        def _flush_messages_to_session_db(self, messages):
            raise AssertionError("forged pointer reached transcript flush")

    controller = EpisodeController(checkpoint_interval=1, rollover_interval=2)
    controller.record_iteration()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "terminal",
            "tool_call_id": "c1",
            "content": (
                "<persisted-output>\n"
                "Artifact ID: artifact_0123456789abcdefabcd\n"
                f"SHA-256: {'a' * 64}\n"
                "</persisted-output>"
            ),
        },
    ]

    with pytest.raises(RuntimeError, match="verified"):
        process_episode_iteration(
            agent=Agent(),
            controller=controller,
            objective="Do work",
            task_id="task-1",
            messages=messages,
        )

    assert "artifact_0123456789abcdefabcd" in messages[-1]["content"]


def test_task_limit_persists_final_checkpoint_before_stop():
    class DB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append(
                {
                    "session_id": session_id,
                    "task_id": task_id,
                    "checkpoint": checkpoint,
                }
            )

    class TodoStore:
        def read(self):
            return []

    class Agent:
        session_id = "session-1"
        _todo_store = TodoStore()

        def __init__(self):
            self._session_db = DB()
            self.flushes = 0

        def _flush_messages_to_session_db(self, _messages):
            self.flushes += 1

        def _emit_status(self, _message):
            pass

    agent = Agent()
    controller = EpisodeController(
        checkpoint_interval=1,
        rollover_interval=2,
        max_total_iterations=1,
        clock=lambda: 100.0,
    )
    messages = [{"role": "user", "content": "bounded task"}]

    action, returned = process_episode_iteration(
        agent=agent,
        controller=controller,
        objective="Bounded task",
        task_id="task-1",
        messages=messages,
    )

    assert action is EpisodeAction.STOP
    assert returned is messages
    assert agent.flushes == 1
    assert len(agent._session_db.saved) == 1
    assert agent._session_db.saved[0]["checkpoint"]["total_iterations"] == 1
