"""Strict JSON input/output helpers for benchmark evidence."""

from __future__ import annotations

from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from paretopilot.domain import BenchmarkSet, Constraints, ValidationError


def _reject_json_constant(value: str) -> None:
    """Reject the non-standard constants accepted by Python's JSON decoder."""

    raise ValidationError(f"non-standard JSON constant {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate keys at every depth."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _json_compatible(
    value: Any,
    *,
    location: str = "$",
    active_containers: set[int] | None = None,
) -> Any:
    """Return built-in JSON-compatible values or raise a contextual error.

    Tuples are normalized to arrays because dataclass ``asdict`` preserves tuple
    fields and Python's JSON encoder already treats them as arrays.
    """

    if active_containers is None:
        active_containers = set()

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{location} must not contain NaN or Infinity")
        return value

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_containers:
            raise ValidationError(f"{location} contains a cyclic object")
        active_containers.add(container_id)
        try:
            normalized: dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValidationError(f"{location} contains a non-string object key")
                normalized[key] = _json_compatible(
                    child,
                    location=f"{location}.{key}",
                    active_containers=active_containers,
                )
            return normalized
        finally:
            active_containers.remove(container_id)

    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            raise ValidationError(f"{location} contains a cyclic array")
        active_containers.add(container_id)
        try:
            return [
                _json_compatible(
                    child,
                    location=f"{location}[{index}]",
                    active_containers=active_containers,
                )
                for index, child in enumerate(value)
            ]
        finally:
            active_containers.remove(container_id)

    raise ValidationError(
        f"{location} contains unsupported JSON value type {type(value).__name__!r}"
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"file must contain UTF-8 text: {path}") from exc
    except OSError as exc:
        raise ValidationError(f"could not read {path}: {exc}") from exc

    try:
        raw = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        normalized = _json_compatible(raw)
    except ValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid JSON value in {path}: {exc}") from exc

    if not isinstance(normalized, Mapping):
        raise ValidationError(f"top-level value in {path} must be an object")
    return normalized


def load_json_object(path: Path) -> Mapping[str, Any]:
    """Load one strict JSON object without applying a domain schema.

    This is the shared entry point for small metadata documents such as
    benchmark settings and generated summaries.  It retains the same UTF-8,
    duplicate-key, finite-number, and top-level-object checks used by the
    benchmark and constraint loaders.
    """

    return _load_json(path)


def load_benchmarks(path: Path) -> BenchmarkSet:
    raw = _load_json(path)
    try:
        return BenchmarkSet.from_mapping(raw)
    except ValidationError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid benchmark data in {path}: {exc}") from exc


def load_constraints(path: Path) -> Constraints:
    raw = _load_json(path)
    try:
        return Constraints.from_mapping(raw)
    except ValidationError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid constraints in {path}: {exc}") from exc


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write strict JSON, refusing existing destinations by default."""

    try:
        normalized = _json_compatible(payload)
        serialized = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except ValidationError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"payload cannot be encoded as JSON: {exc}") from exc

    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and path.exists():
            raise ValidationError(f"refusing to overwrite existing file: {path}")

        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            raise

        if overwrite:
            os.replace(temporary_path, path)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise ValidationError(f"refusing to overwrite existing file: {path}") from exc
            temporary_path.unlink()
            temporary_path = None
    except ValidationError:
        raise
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"could not write {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file with normalized I/O errors."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()
