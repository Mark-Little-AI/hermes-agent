import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.artifact_store import (
    ArtifactIntegrityError,
    ArtifactQuotaExceeded,
    FilesystemArtifactStore,
)


def _quota_race_worker(root, quota, barrier, queue, call_id):
    store = FilesystemArtifactStore(
        root=Path(root), max_artifact_bytes=10_000, quota_bytes=quota, min_free_bytes=0
    )
    barrier.wait()
    try:
        store.put_text("x" * 600 + call_id, tool_name="terminal", tool_call_id=call_id)
    except ArtifactQuotaExceeded:
        queue.put("quota")
    else:
        queue.put("stored")


def test_store_writes_immutable_metadata_and_occurrence_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore()
    content = "competitor data\n" * 20_000

    record = store.put_text(
        content,
        tool_name="mcp__apify__get_dataset_items",
        tool_call_id="call_123",
        session_id="session_456",
    )

    assert record.artifact_id.startswith("artifact_")
    assert Path(record.content_path).read_text(encoding="utf-8") == content

    metadata = json.loads(Path(record.metadata_path).read_text(encoding="utf-8"))
    assert metadata == {
        "artifact_id": record.artifact_id,
        "sha256": record.sha256,
        "size_bytes": len(content.encode("utf-8")),
        "size_chars": len(content),
    }

    reference = json.loads(Path(record.reference_path).read_text(encoding="utf-8"))
    assert reference["artifact_id"] == record.artifact_id
    assert reference["tool_name"] == "mcp__apify__get_dataset_items"
    assert reference["tool_call_id"] == "call_123"
    assert reference["session_id"] == "session_456"
    assert reference["created_at"] == record.created_at


def test_identical_content_deduplicates_with_distinct_references(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore()

    first = store.put_text("same raw result", tool_name="terminal", tool_call_id="one")
    second = store.put_text("same raw result", tool_name="terminal", tool_call_id="two")

    assert second.artifact_id == first.artifact_id
    assert second.content_path == first.content_path
    assert second.reference_path != first.reference_path
    assert Path(first.content_path).read_text(encoding="utf-8") == "same raw result"
    assert json.loads(Path(first.reference_path).read_text())["tool_call_id"] == "one"
    assert json.loads(Path(second.reference_path).read_text())["tool_call_id"] == "two"


def test_unicode_round_trips_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore()
    content = "日本語とemoji 🚀\n" * 1_000

    record = store.put_text(content, tool_name="read_api", tool_call_id="unicode")

    assert Path(record.content_path).read_text(encoding="utf-8") == content
    assert record.size_chars == len(content)
    assert record.size_bytes == len(content.encode("utf-8"))


def test_profiles_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    first = FilesystemArtifactStore().put_text("shared", tool_name="terminal", tool_call_id="a")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
    second = FilesystemArtifactStore().put_text("shared", tool_name="terminal", tool_call_id="b")

    assert first.artifact_id == second.artifact_id
    assert first.content_path != second.content_path
    assert str(tmp_path / "profile-a") in first.content_path
    assert str(tmp_path / "profile-b") in second.content_path


def test_empty_content_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    store = FilesystemArtifactStore()

    with pytest.raises(ValueError, match="empty"):
        store.put_text("", tool_name="terminal", tool_call_id="empty")


def test_existing_tampered_content_is_rejected(tmp_path):
    content = "expected"
    digest = hashlib.sha256(content.encode()).hexdigest()
    artifact_dir = tmp_path / "artifacts" / "sha256" / digest[:2] / digest
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "content.txt").write_text("tampered", encoding="utf-8")

    store = FilesystemArtifactStore(root=tmp_path / "artifacts")
    with pytest.raises(ArtifactIntegrityError, match="integrity"):
        store.put_text(content, tool_name="terminal", tool_call_id="tampered")


def test_existing_symlink_is_rejected(tmp_path):
    content = "expected"
    digest = hashlib.sha256(content.encode()).hexdigest()
    artifact_dir = tmp_path / "artifacts" / "sha256" / digest[:2] / digest
    artifact_dir.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text(content, encoding="utf-8")
    (artifact_dir / "content.txt").symlink_to(target)

    store = FilesystemArtifactStore(root=tmp_path / "artifacts")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        store.put_text(content, tool_name="terminal", tool_call_id="symlink")


def test_artifact_size_and_store_quota_are_enforced(tmp_path):
    size_limited = FilesystemArtifactStore(
        root=tmp_path / "size-limited", max_artifact_bytes=10, quota_bytes=100
    )
    with pytest.raises(ArtifactQuotaExceeded, match="maximum"):
        size_limited.put_text("x" * 11, tool_name="terminal", tool_call_id="large")

    quota_limited = FilesystemArtifactStore(
        root=tmp_path / "quota-limited", max_artifact_bytes=100, quota_bytes=600,
        min_free_bytes=0,
    )
    quota_limited.put_text("a" * 10, tool_name="terminal", tool_call_id="first")
    with pytest.raises(ArtifactQuotaExceeded, match="quota"):
        quota_limited.put_text("b" * 10, tool_name="terminal", tool_call_id="second")


def test_new_directory_entries_are_fsynced(tmp_path, monkeypatch):
    synced = []
    original = FilesystemArtifactStore._fsync_directory

    def spy(path):
        synced.append(Path(path))
        return original(path)

    monkeypatch.setattr(FilesystemArtifactStore, "_fsync_directory", staticmethod(spy))
    root = tmp_path / "nested" / "profile" / "artifacts"
    record = FilesystemArtifactStore(root=root, min_free_bytes=0).put_text(
        "durable", tool_name="terminal", tool_call_id="fsync"
    )
    digest_dir = Path(record.content_path).parent

    assert root.parent in synced
    assert root in synced
    assert root / "sha256" in synced
    assert digest_dir.parent in synced
    assert digest_dir in synced
    assert root / "references" in synced




def test_malformed_profile_config_falls_back_to_safe_defaults(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("artifacts: [", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))

    store = FilesystemArtifactStore()

    assert store.max_artifact_bytes == 256 * 1024 * 1024
    assert store.quota_bytes == 5 * 1024 * 1024 * 1024
    assert store.min_free_bytes == 1024 * 1024 * 1024


def test_atomic_persistence_works_without_posix_fchmod(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "fchmod")
    store = FilesystemArtifactStore(root=tmp_path / "artifacts", min_free_bytes=0)

    record = store.put_text("windows-compatible", tool_name="terminal", tool_call_id="win")

    assert Path(record.content_path).read_text(encoding="utf-8") == "windows-compatible"


def test_private_permissions(tmp_path):
    record = FilesystemArtifactStore(root=tmp_path / "artifacts", min_free_bytes=0).put_text(
        "private", tool_name="terminal", tool_call_id="permissions"
    )
    assert Path(record.content_path).stat().st_mode & 0o777 == 0o600
    assert Path(record.content_path).parent.stat().st_mode & 0o777 == 0o700


def test_limits_load_from_profile_config_yaml(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "artifacts:\n  max_artifact_bytes: 1234\n  quota_bytes: 5678\n  min_free_bytes: 90\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))

    store = FilesystemArtifactStore()

    assert store.max_artifact_bytes == 1234
    assert store.quota_bytes == 5678
    assert store.min_free_bytes == 90


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-lock regression")
def test_cross_process_near_quota_race_allows_only_one_writer(tmp_path):
    root = tmp_path / "artifacts"
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    quota = 1_350
    processes = [
        ctx.Process(target=_quota_race_worker, args=(str(root), quota, barrier, queue, f"call-{i}"))
        for i in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    outcomes = sorted(queue.get(timeout=2) for _ in processes)
    assert outcomes == ["quota", "stored"]
    assert sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) <= quota


def test_concurrent_deduplication_preserves_integrity_and_each_reference(tmp_path):
    store = FilesystemArtifactStore(root=tmp_path / "artifacts")

    def write(index):
        return store.put_text(
            "concurrent content",
            tool_name="terminal",
            tool_call_id=f"call-{index}",
            session_id=f"session-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(write, range(20)))

    assert len({record.artifact_id for record in records}) == 1
    assert len({record.reference_path for record in records}) == 20
    assert Path(records[0].content_path).read_text() == "concurrent content"
    assert {
        json.loads(Path(record.reference_path).read_text())["tool_call_id"]
        for record in records
    } == {f"call-{index}" for index in range(20)}
