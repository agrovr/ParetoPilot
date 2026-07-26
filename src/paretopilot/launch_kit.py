"""Create a safe synthetic starter project for the ParetoPilot decision gate."""

from __future__ import annotations

import hashlib
from importlib import resources
import os
from pathlib import Path, PurePosixPath
import shlex
from tempfile import TemporaryDirectory
from typing import Mapping

from paretopilot.analysis import recommend
from paretopilot.domain import ValidationError
from paretopilot.io import load_benchmarks, load_constraints, write_text


_ACTION_REF = "agrovr/ParetoPilot@db9ccaf37e3c7e807832652e237de813675ed807"
_EXPECTED_SELECTED_ID = "q4-kleidiai"
_TEMPLATE_FILES = (
    (PurePosixPath(".github/workflows/paretopilot.yml"), "paretopilot.yml"),
    (PurePosixPath(".gitignore"), "gitignore.txt"),
    (PurePosixPath("README.md"), "README.md"),
    (PurePosixPath("benchmarks/benchmark-set.json"), "benchmark-set.json"),
    (PurePosixPath("constraints/deployment.json"), "deployment.json"),
)


def create_launch_kit(directory: Path) -> Mapping[str, object]:
    """Create a new synthetic starter project without merging or overwriting files."""

    destination = _resolve_new_destination(directory)
    templates = _load_validated_templates()

    claimed_destination = False
    try:
        destination.mkdir(exist_ok=False)
        claimed_destination = True
        for relative_directory in _required_directories():
            target_directory = destination.joinpath(*relative_directory.parts)
            target_directory.mkdir(exist_ok=False)

        for relative_path, _resource_name in _TEMPLATE_FILES:
            target = destination.joinpath(*relative_path.parts)
            text = templates[relative_path].decode("utf-8")
            expected_sha256 = hashlib.sha256(templates[relative_path]).hexdigest()
            write_text(target, text)
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256:
                raise ValidationError(f"starter-project file changed while being written: {target}")
    except ValidationError as exc:
        suffix = (
            f"; incomplete starter project remains at {destination}"
            if claimed_destination and os.path.lexists(destination)
            else ""
        )
        raise ValidationError(f"could not create starter project: {exc}{suffix}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        suffix = (
            f"; incomplete starter project remains at {destination}"
            if claimed_destination and os.path.lexists(destination)
            else ""
        )
        raise ValidationError(f"could not create starter project: {exc}{suffix}") from exc

    return {
        "schema_version": "1.0",
        "valid": True,
        "classification": "synthetic-launch-kit",
        "synthetic_source": True,
        "selected_id": _EXPECTED_SELECTED_ID,
        "action_ref": _ACTION_REF,
        "output_directory": str(destination),
        "working_directory": str(destination),
        "files": sorted(relative.as_posix() for relative, _resource in _TEMPLATE_FILES),
        "next_command": _next_command(destination),
    }


def _resolve_new_destination(directory: Path) -> Path:
    requested = Path(directory)
    try:
        if os.path.lexists(requested):
            raise ValidationError(
                f"refusing to overwrite existing starter-project destination: {requested}"
            )
        parent = requested.parent.resolve(strict=True)
        if _is_link_or_junction(requested.parent) or not parent.is_dir():
            raise ValidationError("starter-project parent must be an existing real directory")
        if not requested.name or requested.name in {".", ".."}:
            raise ValidationError("starter-project destination must name a new directory")
        destination = parent / requested.name
        if os.path.lexists(destination):
            raise ValidationError(
                f"refusing to overwrite existing starter-project destination: {destination}"
            )
        return destination
    except ValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(
            f"could not resolve starter-project destination {requested}: {exc}"
        ) from exc


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction is not None and is_junction(path))


def _next_command(destination: Path) -> str:
    def command_path(path: Path) -> str:
        if os.name == "nt":
            return f'"{path.as_posix()}"'
        return shlex.quote(str(path))

    return (
        f"paretopilot ci-gate "
        f"{command_path(destination / 'benchmarks' / 'benchmark-set.json')} "
        f"--constraints {command_path(destination / 'constraints' / 'deployment.json')} "
        f"--output-dir {command_path(destination / 'paretopilot-output')} "
        "--allow-synthetic "
        "--expect-selected-id q4-kleidiai"
    )


def _load_templates() -> Mapping[PurePosixPath, bytes]:
    try:
        root = resources.files("paretopilot").joinpath("templates", "launch-kit")
        return {
            relative_path: root.joinpath(resource_name).read_bytes()
            for relative_path, resource_name in _TEMPLATE_FILES
        }
    except (OSError, TypeError) as exc:
        raise ValidationError(f"could not load bundled starter-project templates: {exc}") from exc


def _load_validated_templates() -> Mapping[PurePosixPath, bytes]:
    try:
        templates = _load_templates()
        _validate_templates(templates)
        return templates
    except ValidationError:
        raise
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
        raise ValidationError(
            f"could not validate bundled starter-project templates: {exc}"
        ) from exc


def _validate_templates(templates: Mapping[PurePosixPath, bytes]) -> None:
    expected_paths = {relative for relative, _resource in _TEMPLATE_FILES}
    if set(templates) != expected_paths:
        raise ValidationError("bundled starter-project template set is incomplete")

    for relative_path, data in templates.items():
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
            raise ValidationError(f"unsafe starter-project destination path: {relative_path}")
        if not isinstance(data, bytes):
            raise ValidationError(f"starter-project template must be bytes: {relative_path}")
        if b"\r" in data or not data.endswith(b"\n"):
            raise ValidationError(
                f"starter-project template must use UTF-8 LF text with a trailing newline: "
                f"{relative_path}"
            )
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                f"starter-project template must contain UTF-8 text: {relative_path}"
            ) from exc

    with TemporaryDirectory(prefix="paretopilot-launch-kit-validation-") as directory:
        root = Path(directory)
        for relative_path, data in templates.items():
            path = root.joinpath(*relative_path.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        benchmarks = load_benchmarks(root / "benchmarks" / "benchmark-set.json")
        constraints = load_constraints(root / "constraints" / "deployment.json")
        decision = recommend(benchmarks, constraints)
        if benchmarks.synthetic is not True:
            raise ValidationError(
                "starter-project benchmark fixture must remain explicitly synthetic"
            )
        if decision.get("selected_id") != _EXPECTED_SELECTED_ID:
            raise ValidationError(
                "starter-project fixture no longer produces the documented synthetic decision"
            )


def _required_directories() -> tuple[PurePosixPath, ...]:
    directories: set[PurePosixPath] = set()
    for relative_path, _resource in _TEMPLATE_FILES:
        current = PurePosixPath()
        for part in relative_path.parent.parts:
            current /= part
            if current.parts:
                directories.add(current)
    return tuple(sorted(directories, key=lambda path: (len(path.parts), path.as_posix())))
