"""Behavior contracts for incremental tool-call persistence (#49045).

A destructive or process-terminating tool that runs during tool execution
must not lose the just-executed assistant(tool_calls) block or the tool
results that were produced before it fired.  These tests pin the contract:

    1. run_conversation flushes the assistant tool-call turn to the session
       DB BEFORE handing control to _execute_tool_calls (so a tool that
       restarts/kills the process never orphans the tool-call block).
    2. The SEQUENTIAL tool path flushes each tool result to the session DB
       immediately after appending it — BEFORE the next tool dispatches.
    3. The CONCURRENT tool path flushes each tool result in append order.

These exercise the REAL production dispatch surfaces:

    * sequential -> ``run_agent.handle_function_call`` (tool_executor ~1256/1298)
    * concurrent -> ``agent._invoke_tool`` (tool_executor ~539)

Mocking the genuine dispatch surface keeps the tests deterministic (no real
``web_search`` / network) AND mutation-survivable: the ordering assertions
read snapshots captured at flush time, so removing any production flush call
makes the corresponding assertion fail.
"""

import copy
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from agent.tool_dispatch_helpers import make_tool_result_message
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


# ---------------------------------------------------------------------------
# Contract 1: run_conversation persists the assistant tool-call block BEFORE
# tool execution begins.
# ---------------------------------------------------------------------------
def test_run_conversation_flushes_assistant_tool_call_before_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]

    # Record a deep snapshot of the message list at every flush so the
    # assertion does not depend on later mutations.
    flush_snapshots: list[list] = []

    def _record_flush(messages, conversation_history=None):
        flush_snapshots.append(copy.deepcopy(messages))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    # Capture observations at execute time into module-level lists rather than
    # asserting inside _execute_tool_calls — run_conversation's outer loop
    # swallows exceptions, so an in-callback assertion would never surface.
    executed = {"count": 0}
    snapshot_at_execute: list = []

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        executed["count"] += 1
        # Record the DB state observed at the moment tool execution begins.
        snapshot_at_execute.append(
            copy.deepcopy(flush_snapshots[-1]) if flush_snapshots else None
        )
        # Simulate the tool producing a result (as the real path would).
        messages.append(make_tool_result_message("web_search", "search result", "c1"))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("search something")

    assert executed["count"] == 1, "_execute_tool_calls was never reached"
    # The assistant tool-call block MUST have been flushed before execution.
    last = snapshot_at_execute[0]
    assert last is not None, "no flush occurred before tool execution"
    assert last[-1]["role"] == "assistant"
    assert last[-1]["tool_calls"][0]["id"] == "c1"
    assert result["final_response"] == "done"


def test_run_conversation_rolls_over_to_structured_checkpoint_context():
    agent = _make_agent()
    first_call = _mock_tool_call(call_id="c1")
    second_call = _mock_tool_call(call_id="c2")
    third_call = _mock_tool_call(call_id="c3")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[first_call]),
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[second_call]),
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[third_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]

    class CheckpointDB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))

    checkpoint_db = CheckpointDB()
    agent._session_db = checkpoint_db

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        call = assistant_message.tool_calls[0]
        messages.append(
            make_tool_result_message(
                "web_search", f"raw-result-{call.id}", call.id
            )
        )

    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 2,
            "rollover_interval": 3,
            "max_episodes": 3,
            "max_total_iterations": 6,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("perform three searches")

    assert result["final_response"] == "done"
    assert len(checkpoint_db.saved) == 2
    assert [entry[2]["total_iterations"] for entry in checkpoint_db.saved] == [2, 3]
    assert all(entry[0] == checkpoint_db.saved[0][0] for entry in checkpoint_db.saved)
    fourth_request = str(agent.client.chat.completions.create.call_args_list[3])
    assert "BOUNDED EPISODE CONTINUATION" in fourth_request
    assert "Artifact ID" in fourth_request
    assert "raw-result-c1" not in fourth_request
    assert "raw-result-c2" not in fourth_request
    assert "raw-result-c3" not in fourth_request


def test_rollover_applies_before_retry_after_invalid_tool_transcript():
    agent = _make_agent()
    invalid_one = _mock_tool_call(
        name="nonexistent_tool", call_id="bad1", arguments='{"secret_argument":"one"}'
    )
    invalid_two = _mock_tool_call(
        name="nonexistent_tool", call_id="bad2", arguments='{"secret_argument":"two"}'
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[invalid_one]),
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[invalid_two]),
        _mock_response(content="recovered", finish_reason="stop"),
    ]

    class CheckpointDB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))
            return len(self.saved)

    agent._session_db = CheckpointDB()
    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 2,
            "max_episodes": 3,
            "max_total_iterations": 5,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
    ):
        result = agent.run_conversation("recover from invalid tools")

    assert result["final_response"] == "recovered"
    assert agent.client.chat.completions.create.call_count == 3
    third_request = str(agent.client.chat.completions.create.call_args_list[2])
    assert "BOUNDED EPISODE CONTINUATION" in third_request
    assert "Artifact ID" in third_request
    assert "secret_argument" not in third_request


def test_rollover_applies_before_retry_after_invalid_json_recovery_transcript():
    agent = _make_agent()
    broken_calls = [
        _mock_tool_call(name="web_search", call_id=f"json{index}", arguments="{]")
        for index in range(1, 4)
    ]
    agent.client.chat.completions.create.side_effect = [
        *[
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[call])
            for call in broken_calls
        ],
        _mock_response(content="recovered", finish_reason="stop"),
    ]

    class CheckpointDB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))
            return len(self.saved)

    agent._session_db = CheckpointDB()
    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 3,
            "max_episodes": 3,
            "max_total_iterations": 6,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
    ):
        result = agent.run_conversation("recover from malformed tool arguments")

    assert result["final_response"] == "recovered"
    assert agent.client.chat.completions.create.call_count == 4
    fourth_request = str(agent.client.chat.completions.create.call_args_list[3])
    assert "BOUNDED EPISODE CONTINUATION" in fourth_request
    assert "Artifact ID" in fourth_request
    assert "{]" not in fourth_request


def test_rollover_applies_after_tool_execution_exception_before_retry():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="ok1")],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    call_id="boom2", arguments='{"sentinel":"must-not-cross"}'
                )
            ],
        ),
        _mock_response(content="recovered", finish_reason="stop"),
    ]

    class CheckpointDB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))
            return len(self.saved)

    agent._session_db = CheckpointDB()

    def _execute(assistant_message, messages, effective_task_id, api_call_count):
        call = assistant_message.tool_calls[0]
        if api_call_count == 2:
            raise RuntimeError("tool exploded")
        messages.append(make_tool_result_message("web_search", "raw result", call.id))

    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 2,
            "max_episodes": 3,
            "max_total_iterations": 5,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
    ):
        result = agent.run_conversation("recover after tool execution failure")

    assert result["final_response"] == "recovered"
    assert agent.client.chat.completions.create.call_count == 3
    third_request = str(agent.client.chat.completions.create.call_args_list[2])
    assert "BOUNDED EPISODE CONTINUATION" in third_request
    assert "Artifact ID" in third_request
    assert "must-not-cross" not in third_request
    assert "tool exploded" not in third_request


def test_outer_loop_guard_rolls_over_before_length_continuation_request():
    agent = _make_agent()
    partial = "bounded partial " + ("x" * 5_000) + "tail-must-not-cross"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="seed1")],
        ),
        _mock_response(content=partial, finish_reason="length"),
        _mock_response(content="recovered", finish_reason="stop"),
    ]

    class CheckpointDB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))
            return len(self.saved)

    agent._session_db = CheckpointDB()

    def _execute(assistant_message, messages, effective_task_id, api_call_count):
        call = assistant_message.tool_calls[0]
        messages.append(make_tool_result_message("web_search", "seed result", call.id))

    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 2,
            "max_episodes": 3,
            "max_total_iterations": 5,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
    ):
        result = agent.run_conversation("recover after truncated output")

    assert result["final_response"] == "recovered"
    assert agent.client.chat.completions.create.call_count == 3
    third_request = str(agent.client.chat.completions.create.call_args_list[2])
    assert "BOUNDED EPISODE CONTINUATION" in third_request
    assert "tail-must-not-cross" not in third_request
    assert "finish_reason': 'length'" not in third_request


def test_bounded_episode_counts_retry_and_fallback_provider_attempts(monkeypatch):
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[]),
        _mock_response(content="would otherwise succeed", finish_reason="stop"),
    ]

    class CheckpointDB:
        def __init__(self):
            self.saved = []

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.saved.append((session_id, task_id, checkpoint))
            return len(self.saved)

    checkpoint_db = CheckpointDB()
    agent._session_db = checkpoint_db
    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 3,
            "max_episodes": 3,
            "max_total_iterations": 2,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch("agent.conversation_loop.jittered_backoff", return_value=0),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
    ):
        result = agent.run_conversation("exercise provider retry accounting")

    assert result["error"] == "bounded_episode_task_limit"
    assert agent.client.chat.completions.create.call_count == 2
    assert [entry[2]["total_iterations"] for entry in checkpoint_db.saved] == [1, 2]


def test_bounded_episode_stop_fails_closed_when_checkpoint_persistence_fails():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="discarded at safety limit", finish_reason="stop"),
        _mock_response(content="must never run", finish_reason="stop"),
    ]

    class FailingCheckpointDB:
        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            raise OSError("checkpoint storage unavailable")

    agent._session_db = FailingCheckpointDB()
    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 3,
            "max_episodes": 3,
            "max_total_iterations": 1,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
    ):
        result = agent.run_conversation("stop safely if checkpointing fails")

    assert result["error"] == "bounded_episode_task_limit"
    assert "could not be persisted" in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 1


def test_bounded_episode_rollover_fails_closed_when_checkpoint_persistence_fails():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="c1")],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="c2")],
        ),
        _mock_response(content="must never run", finish_reason="stop"),
    ]

    class FailingRolloverDB:
        def __init__(self):
            self.calls = 0

        def save_task_checkpoint(self, session_id, task_id, checkpoint):
            self.calls += 1
            if self.calls >= 2:
                raise OSError("rollover checkpoint unavailable")
            return self.calls

    agent._session_db = FailingRolloverDB()

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count):
        call = assistant_message.tool_calls[0]
        messages.append(make_tool_result_message("web_search", "raw result", call.id))

    episode_config = {
        "bounded_episodes": {
            "enabled": True,
            "checkpoint_interval": 1,
            "rollover_interval": 2,
            "max_episodes": 3,
            "max_total_iterations": 5,
            "max_wall_seconds": 600,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=episode_config),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("stop safely if rollover cannot persist")

    assert result["error"] == "bounded_episode_rollover_failed"
    assert "existing context was retained" in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Contract 2: the SEQUENTIAL path flushes each tool result immediately, BEFORE
# the next tool dispatches.  Dispatch goes through run_agent.handle_function_call
# (the real production surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_sequential_flushes_each_tool_result_before_next_dispatch():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    # Ordered event log interleaving real dispatches and DB flushes.
    events: list = []

    def _fake_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # The result for call N must have been flushed before call N+1 fires.
        events.append(("dispatch", kwargs.get("tool_call_id")))
        return f"result-{kwargs.get('tool_call_id')}"

    def _record_flush(flush_messages, conversation_history=None):
        # Snapshot the tail tool result that triggered this flush.
        tail = flush_messages[-1]
        events.append(("flush", tail.get("role"), tail.get("tool_call_id")))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch("run_agent.handle_function_call", side_effect=_fake_dispatch) as disp,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # The mock proves we exercised the REAL sequential dispatch surface.
    assert disp.call_count == 2, "sequential path did not dispatch via handle_function_call"

    # Both tool results landed, in order.
    assert [m["role"] for m in messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Ordering contract: each tool result is flushed AFTER its own dispatch
    # and BEFORE the next dispatch. Expected interleaving:
    #   dispatch c1 -> flush c1 -> dispatch c2 -> flush c2
    assert events == [
        ("dispatch", "c1"),
        ("flush", "tool", "c1"),
        ("dispatch", "c2"),
        ("flush", "tool", "c2"),
    ]


# ---------------------------------------------------------------------------
# Contract 3: the CONCURRENT path flushes each collected tool result in append
# order.  Dispatch goes through agent._invoke_tool (the real concurrent
# surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_concurrent_flushes_each_tool_result_in_order():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    invoked_ids: list = []

    def _fake_invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        invoked_ids.append(tool_call_id)
        return f"result-{tool_call_id}"

    # Each flush must observe exactly one more tool result than the previous
    # flush, in append order — i.e. the tail tool_call_id sequence is c1, c2.
    flushed_tool_ids: list = []
    flush_lengths: list = []

    def _record_flush(flush_messages, conversation_history=None):
        flushed_tool_ids.append(flush_messages[-1]["tool_call_id"])
        flush_lengths.append(len([m for m in flush_messages if m.get("role") == "tool"]))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch.object(agent, "_invoke_tool", side_effect=_fake_invoke) as inv,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    # Proves the real concurrent dispatch surface was exercised.
    assert inv.call_count == 2, "concurrent path did not dispatch via _invoke_tool"
    assert sorted(invoked_ids) == ["c1", "c2"]

    # Results appended in deterministic order.
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Each tool result was flushed exactly once, in append order, with the
    # running tool count growing by one each time (1 then 2).  Removing either
    # production flush call breaks one of these assertions.
    assert flushed_tool_ids == ["c1", "c2"]
    assert flush_lengths == [1, 2]
