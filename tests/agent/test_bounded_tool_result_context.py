from pathlib import Path

from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, maybe_persist_tool_result


def test_ten_large_tool_results_keep_raw_data_but_bound_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    raw_results = [f"result-{index}:" + (chr(65 + index) * 300_000) for index in range(10)]

    retained = [
        maybe_persist_tool_result(
            content=content,
            tool_name="mcp__apify__get_dataset_items",
            tool_use_id=f"apify_{index}",
            env=None,
            threshold=100_000,
            session_id="session_apify",
        )
        for index, content in enumerate(raw_results)
    ]

    assert all(PERSISTED_OUTPUT_TAG in item for item in retained)
    assert all("Artifact ID: artifact_" in item for item in retained)
    assert sum(map(len, retained)) < 30_000

    artifact_files = sorted((tmp_path / "profile" / "artifacts").rglob("content.txt"))
    assert len(artifact_files) == 10
    assert sum(path.stat().st_size for path in artifact_files) > 3_000_000
    assert {path.read_text(encoding="utf-8") for path in artifact_files} == set(raw_results)
