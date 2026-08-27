"""Bounded retrieval tool for durable raw tool-result artifacts."""

from __future__ import annotations

import json
import re

from agent.redact import redact_sensitive_text
from tools.artifact_store import (
    ArtifactSensitiveContentError,
    ArtifactStoreError,
    FilesystemArtifactStore,
)
from tools.registry import registry


_ARTIFACT_ID_RE = re.compile(r"artifact_[0-9a-f]{20}\Z")
_JSON_SAFE_CONTENT_CHARS = 8_000


def artifact_get(
    artifact_id: str,
    offset: int = 1,
    limit: int = 500,
    char_offset: int = 0,
) -> str:
    """Retrieve a verified line range without replaying the full artifact."""
    if not isinstance(artifact_id, str) or _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
        return json.dumps({"error": "invalid artifact ID"})
    try:
        result = FilesystemArtifactStore().read_lines(
            artifact_id,
            offset=offset,
            limit=limit,
            char_offset=char_offset,
            max_chars=_JSON_SAFE_CONTENT_CHARS,
        )
        result["content"] = redact_sensitive_text(
            result["content"], file_read=True, force=True
        )
        return json.dumps(result, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except FileNotFoundError:
        return json.dumps({"error": "artifact not found"})
    except ArtifactSensitiveContentError:
        return json.dumps(
            {"error": "artifact contains multiline secret material; retrieval is blocked"}
        )
    except ArtifactStoreError:
        return json.dumps({"error": "artifact integrity or storage error"})
    except OSError:
        return json.dumps({"error": "artifact could not be read safely"})


ARTIFACT_GET_SCHEMA = {
    "name": "artifact_get",
    "description": (
        "Retrieve a verified, bounded section of a durable raw tool-result artifact by its "
        "artifact_<id>. Use this when a persisted-output summary says the full result is stored "
        "as an artifact. Returns line-numbered text plus next_offset and "
        "next_char_offset continuation values; never replays the entire artifact "
        "automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "maxLength": 29,
                "pattern": "^artifact_[0-9a-f]{20}$",
                "description": "Exact artifact ID, for example artifact_0123456789abcdefabcd.",
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": "One-indexed line at which to start.",
            },
            "char_offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": (
                    "Character offset within the first requested line; use the returned "
                    "next_char_offset to continue a long line."
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "default": 500,
                "description": "Maximum number of lines to return.",
            },
        },
        "required": ["artifact_id"],
    },
}


registry.register(
    name="artifact_get",
    toolset="file",
    schema=ARTIFACT_GET_SCHEMA,
    handler=lambda args, **kw: artifact_get(
        artifact_id=args.get("artifact_id", ""),
        offset=args.get("offset", 1),
        limit=args.get("limit", 500),
        char_offset=args.get("char_offset", 0),
    ),
    emoji="📦",
    max_result_size_chars=60_000,
)
