"""Durable, profile-local storage for raw tool-result artifacts.

Artifacts are content-addressed so repeated results deduplicate without adding
another model tool or database dependency. Immutable content metadata is kept
separate from per-occurrence provenance references.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent.redact import redact_sensitive_text
from hermes_constants import get_config_path, get_hermes_home

try:  # POSIX inter-process quota/integrity lock.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Windows inter-process quota/integrity lock.
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_DEFAULT_MAX_ARTIFACT_BYTES = 256 * _MIB
_DEFAULT_QUOTA_BYTES = 5 * _GIB
_DEFAULT_MIN_FREE_BYTES = 1 * _GIB
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()
_ARTIFACT_ID_RE = re.compile(r"artifact_([0-9a-f]{20})")


class ArtifactStoreError(RuntimeError):
    """Base class for durable artifact failures."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Stored bytes or filesystem shape do not match the claimed artifact."""


class ArtifactSensitiveContentError(ArtifactStoreError):
    """Artifact retrieval is blocked because multiline secret material was detected."""


class ArtifactQuotaExceeded(ArtifactStoreError):
    """The configured artifact size, store quota, or free-space floor was hit."""


@dataclass(frozen=True)
class StoredArtifact:
    artifact_id: str
    sha256: str
    content_path: str
    metadata_path: str
    size_chars: int
    size_bytes: int
    total_lines: int
    contains_multiline_secret: bool


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    sha256: str
    content_path: str
    metadata_path: str
    reference_path: str
    size_chars: int
    size_bytes: int
    tool_name: str
    tool_call_id: str
    session_id: str | None
    created_at: str


def _positive_config_int(value, default: int, *, allow_zero: bool = False) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    minimum = 0 if allow_zero else 1
    return parsed if parsed >= minimum else default


def _load_artifact_limits() -> tuple[int, int, int]:
    """Load non-secret artifact limits from the profile's config.yaml."""
    config_path = get_config_path()
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        config = {}
    section = config.get("artifacts", {}) if isinstance(config, dict) else {}
    if not isinstance(section, dict):
        section = {}
    return (
        _positive_config_int(section.get("max_artifact_bytes"), _DEFAULT_MAX_ARTIFACT_BYTES),
        _positive_config_int(section.get("quota_bytes"), _DEFAULT_QUOTA_BYTES),
        _positive_config_int(
            section.get("min_free_bytes"), _DEFAULT_MIN_FREE_BYTES, allow_zero=True
        ),
    )


def _lock_for(root: Path) -> threading.RLock:
    key = str(root.resolve(strict=False))
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


class FilesystemArtifactStore:
    """Content-addressed raw artifact store rooted in the active profile.

    The store is bounded and fail-closed: reaching the quota does not evict raw
    evidence automatically. Callers can fall back to the existing sandbox or
    bounded inline truncation and surface an explicit storage warning.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_artifact_bytes: int | None = None,
        quota_bytes: int | None = None,
        min_free_bytes: int | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else get_hermes_home() / "artifacts"
        configured_max, configured_quota, configured_min_free = _load_artifact_limits()
        self.max_artifact_bytes = (
            configured_max if max_artifact_bytes is None else max_artifact_bytes
        )
        self.quota_bytes = configured_quota if quota_bytes is None else quota_bytes
        self.min_free_bytes = (
            configured_min_free if min_free_bytes is None else min_free_bytes
        )
        for name, value in (
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("quota_bytes", self.quota_bytes),
            ("min_free_bytes", self.min_free_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self._lock = _lock_for(self.root)

    @classmethod
    def _ensure_private_dir(cls, path: Path) -> None:
        missing: list[Path] = []
        cursor = path
        while not cursor.exists() and not cursor.is_symlink():
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent

        if cursor.is_symlink():
            raise ArtifactIntegrityError(f"artifact directory is a symlink: {cursor}")
        if cursor.exists() and not cursor.is_dir():
            raise ArtifactIntegrityError(f"artifact directory is not a directory: {cursor}")

        for directory in reversed(missing):
            created = False
            try:
                directory.mkdir(mode=0o700, parents=False, exist_ok=False)
                created = True
            except FileExistsError:
                pass
            if directory.is_symlink() or not directory.is_dir():
                raise ArtifactIntegrityError(
                    f"artifact directory was replaced during creation: {directory}"
                )
            os.chmod(directory, 0o700)
            if created:
                cls._fsync_directory(directory.parent)
                cls._fsync_directory(directory)

        if path.is_symlink():
            raise ArtifactIntegrityError(f"artifact directory is a symlink: {path}")
        if not path.is_dir():
            raise ArtifactIntegrityError(f"artifact directory is not a directory: {path}")
        os.chmod(path, 0o700)


    @contextmanager
    def _process_lock(self):
        """Serialize quota checks and writes across gateway/cron processes."""
        lock_path = self.root / ".store.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot safely open artifact lock: {lock_path}") from exc
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                getattr(msvcrt, "locking")(fd, getattr(msvcrt, "LK_LOCK"), 1)
            else:  # pragma: no cover - unsupported platform
                raise ArtifactStoreError("no inter-process artifact locking primitive available")
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                getattr(msvcrt, "locking")(fd, getattr(msvcrt, "LK_UNLCK"), 1)
            os.close(fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(path, flags)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def _atomic_write(cls, path: Path, content: bytes) -> None:
        cls._ensure_private_dir(path.parent)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            cls._fsync_directory(path.parent)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _inspect_utf8_file(path: Path) -> tuple[str, int, int, int, bool]:
        """Stream-verify a regular UTF-8 file without loading it into memory."""
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot safely open artifact content: {path}") from exc
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")()
        size_bytes = 0
        size_chars = 0
        newline_count = 0
        last_byte = None
        secret_tail = b""
        redaction_tail = ""
        contains_multiline_secret = False
        try:
            with os.fdopen(fd, "rb") as handle:
                opened_stat = os.fstat(handle.fileno())
                if not stat_module.S_ISREG(opened_stat.st_mode):
                    raise ArtifactIntegrityError(
                        f"artifact content is not a regular file: {path}"
                    )
                while True:
                    chunk = handle.read(_MIB)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    newline_count += chunk.count(b"\n")
                    last_byte = chunk[-1]
                    probe = (secret_tail + chunk).upper()
                    if b"-----BEGIN" in probe and b"PRIVATE KEY-----" in probe:
                        contains_multiline_secret = True
                    secret_tail = probe[-256:]
                    try:
                        decoded_text = decoder.decode(chunk, final=False)
                        size_chars += len(decoded_text)
                        redaction_probe = redaction_tail + decoded_text
                        if redact_sensitive_text(
                            redaction_probe, force=True, file_read=True
                        ) != redaction_probe:
                            contains_multiline_secret = True
                        redaction_tail = redaction_probe[-4096:]
                    except UnicodeDecodeError as exc:
                        raise ArtifactIntegrityError(
                            f"artifact content is not valid UTF-8: {path}"
                        ) from exc
                try:
                    size_chars += len(decoder.decode(b"", final=True))
                except UnicodeDecodeError as exc:
                    raise ArtifactIntegrityError(
                        f"artifact content is not valid UTF-8: {path}"
                    ) from exc
        except Exception:
            raise
        total_lines = newline_count + (1 if size_bytes and last_byte != 0x0A else 0)
        return (
            digest.hexdigest(), size_bytes, size_chars, total_lines,
            contains_multiline_secret,
        )

    @staticmethod
    def _read_regular_file(path: Path) -> bytes:
        try:
            stat = path.lstat()
        except FileNotFoundError:
            raise
        if path.is_symlink():
            raise ArtifactIntegrityError(f"artifact content path is a symlink: {path}")
        if not path.is_file():
            raise ArtifactIntegrityError(f"artifact content path is not a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot safely open artifact: {path}") from exc
        try:
            with os.fdopen(fd, "rb") as handle:
                return handle.read()
        finally:
            # fdopen owns and closes fd; this branch is only for type clarity.
            del stat

    @classmethod
    def _ensure_exact_file(cls, path: Path, expected: bytes, *, label: str) -> None:
        if path.exists() or path.is_symlink():
            actual = cls._read_regular_file(path)
            if actual != expected:
                raise ArtifactIntegrityError(f"{label} integrity mismatch: {path}")
            return
        cls._atomic_write(path, expected)
        actual = cls._read_regular_file(path)
        if actual != expected:
            raise ArtifactIntegrityError(f"{label} integrity mismatch after write: {path}")

    def _store_size_bytes(self) -> int:
        total = 0
        if not self.root.exists():
            return 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def _check_capacity(self, additional_bytes: int) -> None:
        current = self._store_size_bytes()
        if current + additional_bytes > self.quota_bytes:
            raise ArtifactQuotaExceeded(
                f"artifact store quota exceeded: {current + additional_bytes} > {self.quota_bytes} bytes"
            )
        free = shutil.disk_usage(self.root).free
        if free - additional_bytes < self.min_free_bytes:
            raise ArtifactQuotaExceeded(
                f"artifact store free-space floor reached: {free} bytes free"
            )

    def resolve(self, artifact_id: str) -> StoredArtifact:
        """Resolve an artifact ID and verify content plus immutable metadata."""
        match = _ARTIFACT_ID_RE.fullmatch(artifact_id or "")
        if match is None:
            raise ValueError("invalid artifact ID")
        prefix = match.group(1)
        sha_root = self.root / "sha256"
        prefix_dir = sha_root / prefix[:2]
        for directory in (self.root, sha_root, prefix_dir):
            if directory.is_symlink():
                raise ArtifactIntegrityError("artifact storage path contains a symlink")
            if directory.exists() and not directory.is_dir():
                raise ArtifactIntegrityError("artifact storage path is not a directory")
        if not prefix_dir.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        candidates = [
            path for path in prefix_dir.glob(f"{prefix}*")
            if path.is_dir() and not path.is_symlink() and len(path.name) == 64
        ]
        if not candidates:
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        if len(candidates) != 1:
            raise ArtifactIntegrityError(f"artifact ID is ambiguous: {artifact_id}")

        artifact_dir = candidates[0]
        try:
            artifact_dir.resolve(strict=True).relative_to(self.root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError("artifact resolved outside the configured store") from exc
        digest = artifact_dir.name
        content_path = artifact_dir / "content.txt"
        metadata_path = artifact_dir / "metadata.json"
        (
            actual_digest,
            size_bytes,
            size_chars,
            total_lines,
            contains_multiline_secret,
        ) = self._inspect_utf8_file(content_path)
        if actual_digest != digest:
            raise ArtifactIntegrityError(f"artifact content integrity mismatch: {content_path}")

        try:
            metadata = json.loads(self._read_regular_file(metadata_path).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"artifact metadata integrity mismatch: {metadata_path}") from exc
        expected = {
            "artifact_id": f"artifact_{digest[:20]}",
            "sha256": digest,
            "size_bytes": size_bytes,
            "size_chars": size_chars,
        }
        if metadata != expected:
            raise ArtifactIntegrityError(f"artifact metadata integrity mismatch: {metadata_path}")

        return StoredArtifact(
            artifact_id=expected["artifact_id"],
            sha256=digest,
            content_path=str(content_path),
            metadata_path=str(metadata_path),
            size_chars=size_chars,
            size_bytes=size_bytes,
            total_lines=total_lines,
            contains_multiline_secret=contains_multiline_secret,
        )

    def read_lines(
        self,
        artifact_id: str,
        *,
        offset: int = 1,
        limit: int = 500,
        char_offset: int = 0,
        max_chars: int = 50_000,
    ) -> dict:
        """Return a verified, bounded, line-numbered artifact section."""
        if offset < 1:
            raise ValueError("offset must be at least 1")
        if limit < 1 or limit > 2_000:
            raise ValueError("limit must be between 1 and 2000")
        if char_offset < 0:
            raise ValueError("char_offset must be at least 0")
        if max_chars < 1 or max_chars > 50_000:
            raise ValueError("max_chars must be between 1 and 50000")

        stored = self.resolve(artifact_id)
        if stored.contains_multiline_secret:
            raise ArtifactSensitiveContentError(
                "artifact contains multiline secret material; retrieval is blocked"
            )
        target_end = offset + limit - 1
        rendered: list[str] = []
        used = 0
        truncated_by = None
        line_number = 1
        current_parts: list[str] = []
        current_chars = 0
        current_seen = 0
        current_truncated = False
        continuation_line: int | None = None
        continuation_char: int | None = None
        last_was_newline = False

        def capture(fragment: str) -> None:
            nonlocal current_chars, current_seen, current_truncated
            if not (offset <= line_number <= target_end):
                return
            start_at = char_offset if line_number == offset else 0
            fragment_start = current_seen
            current_seen += len(fragment)
            skip = max(0, start_at - fragment_start)
            if skip >= len(fragment):
                return
            eligible = fragment[skip:]
            remaining_capture = max_chars - current_chars
            if remaining_capture > 0:
                current_parts.append(eligible[:remaining_capture])
                current_chars += min(len(eligible), remaining_capture)
            if len(eligible) > remaining_capture:
                current_truncated = True

        def finish_line() -> None:
            nonlocal line_number, current_parts, current_chars, current_seen
            nonlocal current_truncated, used, truncated_by
            nonlocal continuation_line, continuation_char
            if offset <= line_number <= target_end:
                start_at = char_offset if line_number == offset else 0
                line = "".join(current_parts)
                if line.endswith("\r"):
                    line = line[:-1]
                prefix = f"{line_number}|"
                separator = 1 if rendered else 0
                available = max_chars - used - separator
                content_available = max(0, available - len(prefix))
                returned_content = line[:content_available]
                candidate = prefix + returned_content
                if available > 0:
                    rendered.append(candidate[:available])
                    used += min(len(candidate), available) + separator
                if current_truncated or len(line) > content_available or available <= 0:
                    truncated_by = "character_budget"
                    if continuation_line is None:
                        continuation_line = line_number
                        continuation_char = start_at + len(returned_content)
            line_number += 1
            current_parts = []
            current_chars = 0
            current_seen = 0
            current_truncated = False

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(stored.content_path, flags)
        except OSError as exc:
            raise ArtifactIntegrityError("artifact changed after integrity verification") from exc
        decoder = codecs.getincrementaldecoder("utf-8")()
        second_digest = hashlib.sha256()
        reached_eof = False
        try:
            with os.fdopen(fd, "rb") as handle:
                opened_stat = os.fstat(handle.fileno())
                if not stat_module.S_ISREG(opened_stat.st_mode):
                    raise ArtifactIntegrityError("artifact changed after integrity verification")
                while True:
                    chunk = handle.read(_MIB)
                    if not chunk:
                        reached_eof = True
                        break
                    second_digest.update(chunk)
                    decoded = decoder.decode(chunk, final=False)
                    if line_number > target_end:
                        continue
                    parts = decoded.split("\n")
                    for index, part in enumerate(parts):
                        if index:
                            finish_line()
                            last_was_newline = True
                            if line_number > target_end:
                                break
                        capture(part)
                        if part:
                            last_was_newline = False
                if reached_eof:
                    tail = decoder.decode(b"", final=True)
                    if line_number <= target_end:
                        capture(tail)
        except UnicodeDecodeError as exc:
            raise ArtifactIntegrityError("artifact changed after integrity verification") from exc

        if second_digest.hexdigest() != stored.sha256:
            raise ArtifactIntegrityError("artifact changed after integrity verification")

        if stored.size_bytes and not last_was_newline and line_number <= stored.total_lines:
            finish_line()

        returned_lines = len(rendered)
        next_offset = offset + returned_lines
        result = {
            "artifact_id": stored.artifact_id,
            "sha256": stored.sha256,
            "content": "\n".join(rendered),
            "total_lines": stored.total_lines,
            "size_chars": stored.size_chars,
            "size_bytes": stored.size_bytes,
            "truncated": truncated_by is not None or next_offset <= stored.total_lines,
        }
        if truncated_by is not None:
            result["truncated_by"] = truncated_by
            result["next_offset"] = continuation_line
            result["next_char_offset"] = continuation_char
        elif next_offset <= stored.total_lines:
            result["next_offset"] = next_offset
            result["next_char_offset"] = 0
        return result

    def put_text(
        self,
        content: str,
        *,
        tool_name: str,
        tool_call_id: str,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        if not content:
            raise ValueError("artifact content must not be empty")

        raw = content.encode("utf-8")
        if len(raw) > self.max_artifact_bytes:
            raise ArtifactQuotaExceeded(
                f"artifact exceeds maximum size: {len(raw)} > {self.max_artifact_bytes} bytes"
            )

        digest = hashlib.sha256(raw).hexdigest()
        artifact_id = f"artifact_{digest[:20]}"
        artifact_dir = self.root / "sha256" / digest[:2] / digest
        content_path = artifact_dir / "content.txt"
        metadata_path = artifact_dir / "metadata.json"
        created_at = datetime.now(timezone.utc).isoformat()
        reference_id = uuid.uuid4().hex
        reference_path = self.root / "references" / f"{reference_id}.json"

        metadata = {
            "artifact_id": artifact_id,
            "sha256": digest,
            "size_bytes": len(raw),
            "size_chars": len(content),
        }
        metadata_bytes = json.dumps(
            metadata, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        reference = {
            "artifact_id": artifact_id,
            "created_at": created_at,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        }
        reference_bytes = json.dumps(
            reference, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")

        with self._lock:
            self._ensure_private_dir(self.root)
            with self._process_lock():
                self._ensure_private_dir(self.root / "sha256")
                self._ensure_private_dir(self.root / "sha256" / digest[:2])
                self._ensure_private_dir(artifact_dir)
                self._ensure_private_dir(self.root / "references")

                content_exists = content_path.exists() or content_path.is_symlink()
                metadata_exists = metadata_path.exists() or metadata_path.is_symlink()
                additional = len(reference_bytes)
                if not content_exists:
                    additional += len(raw)
                if not metadata_exists:
                    additional += len(metadata_bytes)
                self._check_capacity(additional)

                self._ensure_exact_file(content_path, raw, label="artifact content")
                self._ensure_exact_file(metadata_path, metadata_bytes, label="artifact metadata")
                self._atomic_write(reference_path, reference_bytes)

        return ArtifactRecord(
            artifact_id=artifact_id,
            sha256=digest,
            content_path=str(content_path),
            metadata_path=str(metadata_path),
            reference_path=str(reference_path),
            size_chars=len(content),
            size_bytes=len(raw),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            session_id=session_id,
            created_at=created_at,
        )
