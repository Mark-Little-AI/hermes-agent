import json
from pathlib import Path

from tools.artifact_store import FilesystemArtifactStore
from tools.artifact_tool import artifact_get
from tools.registry import registry


def test_artifact_get_is_registered_in_file_toolset():
    assert registry.get_toolset_for_tool("artifact_get") == "file"
    schema = registry.get_schema("artifact_get")
    assert schema is not None
    assert schema["name"] == "artifact_get"
    assert schema["parameters"]["required"] == ["artifact_id"]


def test_artifact_get_streams_content_instead_of_full_file_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore(min_free_bytes=0)
    record = store.put_text(
        "line\n" * 400_000,
        tool_name="terminal",
        tool_call_id="streamed",
    )
    original = FilesystemArtifactStore._read_regular_file

    def reject_full_content_read(path):
        if path.name == "content.txt":
            raise AssertionError("artifact content must be streamed")
        return original(path)

    monkeypatch.setattr(
        FilesystemArtifactStore, "_read_regular_file", staticmethod(reject_full_content_read)
    )
    result = json.loads(artifact_get(record.artifact_id, offset=200_000, limit=2))

    assert result["content"].startswith("200000|line")
    assert result["total_lines"] == 400_000


def test_artifact_get_detects_change_between_verification_and_range_read(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore(min_free_bytes=0)
    record = store.put_text("trusted\n", tool_name="terminal", tool_call_id="race")
    original_resolve = FilesystemArtifactStore.resolve

    def mutate_after_resolve(self, artifact_id):
        resolved = original_resolve(self, artifact_id)
        with open(resolved.content_path, "w", encoding="utf-8") as handle:
            handle.write("changed\n")
        return resolved

    monkeypatch.setattr(FilesystemArtifactStore, "resolve", mutate_after_resolve)
    result = json.loads(artifact_get(record.artifact_id))

    assert "integrity" in result["error"]


def test_artifact_get_rejects_parent_directory_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore(min_free_bytes=0)
    record = store.put_text("trusted\n", tool_name="terminal", tool_call_id="parent-link")
    digest_dir = Path(record.content_path).parent
    prefix_dir = digest_dir.parent
    outside = tmp_path / "outside-prefix"
    prefix_dir.rename(outside)
    prefix_dir.symlink_to(outside, target_is_directory=True)

    result = json.loads(artifact_get(record.artifact_id))

    assert "integrity" in result["error"]
    assert str(tmp_path) not in result["error"]


def test_artifact_get_long_line_returns_character_continuation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        "first\n" + "x" * 20_000 + "\nthird\n",
        tool_name="terminal",
        tool_call_id="long-line",
    )

    pages = []
    next_line = 2
    next_char = 0
    while True:
        page = json.loads(
            artifact_get(
                record.artifact_id,
                offset=next_line,
                limit=1,
                char_offset=next_char,
            )
        )
        pages.append(page["content"].split("|", 1)[1])
        if page.get("next_offset") != 2:
            break
        next_line = page["next_offset"]
        next_char = page["next_char_offset"]

    assert len(pages) >= 3
    assert "".join(pages) == "x" * 20_000


def test_artifact_get_serialized_output_stays_within_registry_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        "\0" * 50_000, tool_name="terminal", tool_call_id="escaped"
    )

    serialized = artifact_get(record.artifact_id)

    assert len(serialized) <= 60_000


def test_artifact_get_does_not_reflect_oversized_invalid_id():
    result = artifact_get("x" * 1_000_000)
    assert len(result) < 500
    assert "x" * 100 not in result


def test_artifact_get_force_redacts_even_when_config_disables_redaction(
    tmp_path, monkeypatch
):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "security:\n  redact_secrets: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))
    secret = "Authorization: Bearer super-secret-token-value"
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        secret, tool_name="terminal", tool_call_id="secret"
    )

    result = json.loads(artifact_get(record.artifact_id))

    serialized = json.dumps(result)
    assert "super-secret-token-value" not in serialized
    assert "secret material" in result.get("error", "") or "..." in result.get(
        "content", ""
    )


def test_artifact_get_blocks_multiline_private_key_across_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    content = (
        "before\n-----BEGIN PRIVATE KEY-----\n"
        + "A" * 20_000
        + "\n-----END PRIVATE KEY-----\nafter\n"
    )
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        content, tool_name="terminal", tool_call_id="pem"
    )

    result = json.loads(artifact_get(record.artifact_id, offset=3, limit=1))

    assert "multiline secret" in result["error"]
    assert "AAAA" not in json.dumps(result)


def test_artifact_get_retrieves_bounded_line_range_by_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        "first\nsecond\nthird\nfourth\n",
        tool_name="terminal",
        tool_call_id="source-call",
        session_id="source-session",
    )

    result = json.loads(artifact_get(record.artifact_id, offset=2, limit=2))

    assert result["artifact_id"] == record.artifact_id
    assert result["content"] == "2|second\n3|third"
    assert result["total_lines"] == 4
    assert result["next_offset"] == 4
    assert result["sha256"] == record.sha256


def test_artifact_get_clamps_large_single_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        "x" * 100_000,
        tool_name="mcp__apify__get_dataset_items",
        tool_call_id="large-line",
    )

    result = json.loads(artifact_get(record.artifact_id, offset=1, limit=10))

    assert len(result["content"]) <= 50_000
    assert result["truncated_by"] == "character_budget"
    assert result["total_lines"] == 1


def test_artifact_get_rejects_invalid_or_missing_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))

    invalid = json.loads(artifact_get("../../auth.json"))
    missing = json.loads(artifact_get("artifact_00000000000000000000"))

    assert "invalid artifact ID" in invalid["error"]
    assert "not found" in missing["error"]


def test_artifact_get_detects_tampering_before_return(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    record = FilesystemArtifactStore(min_free_bytes=0).put_text(
        "trusted\n", tool_name="terminal", tool_call_id="trusted"
    )
    with open(record.content_path, "w", encoding="utf-8") as handle:
        handle.write("tampered\n")

    result = json.loads(artifact_get(record.artifact_id))

    assert "integrity" in result["error"]
