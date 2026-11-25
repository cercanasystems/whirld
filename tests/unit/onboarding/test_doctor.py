"""Read-only doctor checks real contracts and checkpoint identities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from whirld import config
from whirld.core.fetch import Manifest
from whirld.core.registry import ModelEntry
from whirld.errors import ModelNotFoundError, SecurityError
from whirld.onboarding import doctor
from whirld.onboarding.doctor import (
    DependencyCheck,
    DiagnosticCheck,
    DiagnosticsService,
    DoctorError,
    DoctorReport,
)


@pytest.fixture
def healthy_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only optional import observation; registry/cache checks remain real."""

    def dependency(distribution: str, module: str) -> DependencyCheck:
        """Return a typed successful dependency snapshot without loading frameworks."""
        return DependencyCheck(
            distribution=distribution,
            module=module,
            version="1.0",
            status="pass",
            message="Fixture runtime imports successfully.",
        )

    monkeypatch.setattr(DiagnosticsService, "_dependency", staticmethod(dependency))
    monkeypatch.setattr(
        doctor.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=2**40),
    )


def service(tmp_path: Path) -> DiagnosticsService:
    """Select a new explicit diagnostic home without mutating user configuration."""
    return DiagnosticsService(config.WhirldPaths(tmp_path / "new-home"))


def fake_entry(payload: bytes) -> ModelEntry:
    """Retain the real RN50 contract with independently hashed tiny fixture weights."""
    entry = ModelEntry.from_registry_file(
        config.bundled_registry_dir() / "models" / "remoteclip-rn50.yaml"
    )
    document = entry.model_dump(exclude_none=True)
    document["distribution"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return ModelEntry.from_registry_data(document)


def install_fixture(
    target: DiagnosticsService, entry: ModelEntry, payload: bytes
) -> Manifest:
    """Write isolated manifest/checkpoint bytes for independent SHA verification."""
    directory = target._paths.model_dir(entry.name)
    directory.mkdir(parents=True)
    filename = entry.source.files[0]
    target._paths.model_file(entry.name, filename).write_bytes(payload)
    manifest = Manifest(
        name=entry.name,
        version=entry.version,
        source_type="huggingface",
        sha256=entry.distribution.sha256,
        weights_file=filename,
        entry_sha256=entry.fingerprint(),
        required_files=[filename],
        artifact_sizes={filename: len(payload)},
    )
    target._paths.model_manifest(entry.name).write_text(manifest.model_dump_json())
    return manifest


def check(report: DoctorReport, code: str) -> DiagnosticCheck:
    """Find a unique report row by its public observation identity."""
    return next(item for item in report.checks if item.code == code)


def test_doctor_module_import_remains_lightweight() -> None:
    """Importing services for CLI registration never imports optional frameworks."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import whirld.onboarding.doctor,sys; "
            "assert not {'torch','open_clip','terratorch'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_weights_are_ready_to_download_without_cache_mutation(
    tmp_path: Path, healthy_dependencies: None
) -> None:
    """A healthy fresh setup proceeds to demo even before its model is cached."""
    target = service(tmp_path)
    report = target.inspect()
    assert report.model == "remoteclip-rn50" and report.device == "cpu"
    assert report.ready_for_demo and not report.model_installed
    assert report.model_download_bytes == report.model_size_bytes == 408397807
    assert report.disk_required_bytes == 2 * 408397807 + 64 * 1024**2
    assert check(report, "model").status == "warn"
    assert check(report, "registry").status == "pass"
    assert not target._paths.home.exists()
    report.require_ready()
    assert DoctorReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("model", ["../escape", "/absolute", "UPPERCASE"])
def test_model_identity_is_validated_before_filesystem_use(
    tmp_path: Path, model: str
) -> None:
    """Invalid identifiers never become cache or bundled-resource path components."""
    with pytest.raises(SecurityError):
        service(tmp_path).inspect(model)
    assert list(tmp_path.iterdir()) == []


def test_unknown_bundled_model_has_actionable_domain_error(tmp_path: Path) -> None:
    """Unknown selection is rejected without registry seeding or framework imports."""
    with pytest.raises(ModelNotFoundError, match="No bundled model"):
        service(tmp_path).inspect("unknown-model")
    assert list(tmp_path.iterdir()) == []


def test_existing_changed_registry_is_reported_without_overwriting(
    tmp_path: Path, healthy_dependencies: None
) -> None:
    """Demo provenance cannot silently use a cached model override."""
    target = service(tmp_path)
    active = target._paths.registry_models_dir / "remoteclip-rn50.yaml"
    active.parent.mkdir(parents=True)
    original = (config.bundled_registry_dir() / "models" / active.name).read_text()
    changed = original.replace("RemoteCLIP (RN50)", "Changed local model")
    active.write_text(changed)
    report = target.inspect()
    assert not report.ready_for_demo and check(report, "registry").status == "fail"
    assert active.read_text() == changed
    assert not target._paths.models_dir.exists()
    with pytest.raises(DoctorError, match="registry"):
        report.require_ready()


def test_corrupt_registry_is_reported_without_repair(
    tmp_path: Path, healthy_dependencies: None
) -> None:
    """Malformed active YAML remains untouched for the user to review."""
    target = service(tmp_path)
    active = target._paths.registry_models_dir / "remoteclip-rn50.yaml"
    active.parent.mkdir(parents=True)
    active.write_text("not: [valid")
    report = target.inspect()
    assert check(report, "registry").status == "fail"
    assert active.read_text() == "not: [valid"


def test_symlinked_active_registry_is_rejected_even_when_bytes_match(
    tmp_path: Path, healthy_dependencies: None
) -> None:
    """Doctor enforces the final-file containment rule used by actual inference."""
    target = service(tmp_path)
    active = target._paths.registry_models_dir / "remoteclip-rn50.yaml"
    active.parent.mkdir(parents=True)
    active.symlink_to(config.bundled_registry_dir() / "models" / active.name)
    report = target.inspect()
    assert check(report, "registry").status == "fail"
    assert not report.ready_for_demo and active.is_symlink()


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((3, 13, 6), "fail"),
        ((3, 13, 7), "pass"),
        ((3, 13, 8), "warn"),
        ((3, 14, 0), "warn"),
    ],
)
def test_python_support_distinguishes_minimum_and_verified_version(
    tmp_path: Path,
    healthy_dependencies: None,
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int, int],
    expected: str,
) -> None:
    """Newer interpreters are unverified warnings; unsupported old versions block."""
    monkeypatch.setattr(doctor.sys, "version_info", version)
    report = service(tmp_path).inspect()
    assert check(report, "python").status == expected
    assert report.ready_for_demo == (expected != "fail")


@pytest.mark.parametrize("free", [0, 100])
def test_insufficient_disk_blocks_before_model_download(
    tmp_path: Path,
    healthy_dependencies: None,
    monkeypatch: pytest.MonkeyPatch,
    free: int,
) -> None:
    """Storage failures retain download size and offer a writable-home remedy."""
    monkeypatch.setattr(
        doctor.shutil, "disk_usage", lambda path: SimpleNamespace(free=free)
    )
    report = service(tmp_path).inspect()
    assert not report.ready_for_demo and check(report, "disk").status == "fail"
    assert report.disk_free_bytes == free
    assert not report.cache_path.exists()


def test_unwritable_cache_is_a_blocker_without_write_probe(
    tmp_path: Path, healthy_dependencies: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission estimates use an existing ancestor and create no directories."""
    observed: list[Path] = []

    def access(path: Path, mode: int) -> bool:
        """Record the real ancestor checked by the permission estimate."""
        observed.append(path)
        return False

    monkeypatch.setattr(doctor.os, "access", access)
    report = service(tmp_path).inspect()
    assert check(report, "cache").status == "fail" and not report.ready_for_demo
    assert observed == [tmp_path]
    assert list(tmp_path.iterdir()) == []


def test_disk_inspection_error_is_reported(
    tmp_path: Path, healthy_dependencies: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable filesystem statistics never masquerade as sufficient space."""

    def unavailable(path: Path) -> None:
        """Simulate a denied filesystem metadata read."""
        raise PermissionError("fixture denied")

    monkeypatch.setattr(doctor.shutil, "disk_usage", unavailable)
    report = service(tmp_path).inspect()
    assert report.disk_free_bytes is None
    assert not report.ready_for_demo and check(report, "disk").status == "fail"


def test_verified_installed_model_is_checked_with_real_hash_and_no_changes(
    tmp_path: Path,
) -> None:
    """The primary checkpoint identity is observed from bytes, not manifest claims."""
    payload = b"independently verified fixture weights"
    target = service(tmp_path)
    entry = fake_entry(payload)
    install_fixture(target, entry, payload)
    before = {
        path: path.read_bytes()
        for path in target._paths.home.rglob("*")
        if path.is_file()
    }
    installed, result = target._model(entry)
    assert installed and result.status == "pass"
    assert before == {
        path: path.read_bytes()
        for path in target._paths.home.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "failure", ["checksum", "truncated", "missing", "manifest", "stale", "auxiliary"]
)
def test_invalid_installed_weights_fail_and_propose_force_pull(
    tmp_path: Path, failure: str
) -> None:
    """Corrupt, incomplete and stale installs require explicit repair."""
    payload = b"checkpoint fixture"
    target = service(tmp_path)
    entry = fake_entry(payload)
    manifest = install_fixture(target, entry, payload)
    path = target._paths.model_file(entry.name, manifest.weights_file)
    if failure == "checksum":
        path.write_bytes(b"x" * len(payload))
    elif failure == "truncated":
        path.write_bytes(b"x")
    elif failure == "missing":
        path.unlink()
    elif failure == "manifest":
        target._paths.model_manifest(entry.name).write_bytes(b"\xff")
    elif failure == "stale":
        target._paths.model_manifest(entry.name).write_text(
            manifest.model_copy(update={"version": "stale"}).model_dump_json()
        )
    else:
        target._paths.model_manifest(entry.name).write_text(
            manifest.model_copy(
                update={"required_files": [manifest.weights_file, "missing.yaml"]}
            ).model_dump_json()
        )
    installed, result = target._model(entry)
    assert not installed and result.status == "fail"
    assert result.remedy is not None and "--force" in result.remedy
    if failure == "checksum":
        assert "SHA-256" in result.message


def test_dependency_metadata_absence_skips_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing distributions fail clearly without an unrelated import attempt."""

    def missing(name: str) -> str:
        """Simulate absent optional runtime metadata."""
        raise importlib.metadata.PackageNotFoundError(name)

    def unexpected(name: str) -> None:
        """Fail if the runtime boundary imports an absent distribution."""
        raise AssertionError("unexpected import")

    monkeypatch.setattr(doctor.importlib.metadata, "version", missing)
    monkeypatch.setattr(doctor.importlib, "import_module", unexpected)
    result = DiagnosticsService._dependency("open_clip_torch", "open_clip")
    assert result.status == "fail" and result.version is None


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError, AttributeError])
def test_broken_dependency_imports_are_diagnosed(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """Native library and incompatible package failures become actionable rows."""

    def broken(name: str) -> None:
        """Simulate a selected runtime with broken binary/package compatibility."""
        raise error("fixture details")

    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(doctor.importlib, "import_module", broken)
    result = DiagnosticsService._dependency("open_clip_torch", "open_clip")
    assert result.status == "fail" and result.version == "1.2.3"
    assert error.__name__ in result.message


def test_actual_selected_dependency_versions_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distribution names are distinct from module names such as open_clip."""
    imports: list[str] = []
    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda name: "3.3.0")
    monkeypatch.setattr(doctor.importlib, "import_module", imports.append)
    result = DiagnosticsService._dependency("open_clip_torch", "open_clip")
    assert result.version == "3.3.0" and result.status == "pass"
    assert imports == ["open_clip"]


def test_linux_missing_expat_reports_os_repair_instead_of_python_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing native XML library needs the distribution runtime package."""
    original = DiagnosticsService._dependency
    original_import = doctor.importlib.import_module
    original_version = doctor.importlib.metadata.version

    def broken(name: str) -> None:
        """Reproduce the import error observed in the Linux slim first-run test."""
        raise ImportError(
            "libexpat.so.1: cannot open shared object file: No such file or directory"
        )

    # Use the actual import observation for this one dependency, then its report row.
    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda name: "1.5.2")
    monkeypatch.setattr(doctor.importlib, "import_module", broken)
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    dependency = original("rasterio", "rasterio")
    assert dependency.status == "fail" and dependency.version == "1.5.2"
    assert dependency.remedy is not None and "Debian/Ubuntu" in dependency.remedy
    assert "apt-get install libexpat1" in dependency.remedy
    assert "pip install" not in dependency.remedy
    monkeypatch.setattr(doctor.importlib, "import_module", original_import)
    monkeypatch.setattr(doctor.importlib.metadata, "version", original_version)

    def observation(package: str, module: str) -> DependencyCheck:
        """Retain the real failed observation alongside healthy selected runtimes."""
        return (
            dependency
            if module == "rasterio"
            else DependencyCheck(
                distribution=package,
                module=module,
                version="1.0",
                status="pass",
                message="Fixture runtime ready.",
            )
        )

    monkeypatch.setattr(DiagnosticsService, "_dependency", staticmethod(observation))
    monkeypatch.setattr(
        doctor.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**40)
    )
    report = service(tmp_path).inspect()
    assert not report.ready_for_demo
    assert check(report, "dependency:rasterio").remedy == dependency.remedy


class PackageDistribution:
    """Represent installed source metadata independently of machine packaging state."""

    version = "0.3.0"

    def __init__(self, origin: str | None) -> None:
        """Bind a PEP 610 document or ordinary-index absence."""
        self.origin = origin

    def read_text(self, name: str) -> str | None:
        """Return only the package-origin metadata requested by doctor."""
        assert name == "direct_url.json"
        return self.origin


@pytest.mark.parametrize(
    ("origin", "target", "editable"),
    [
        (None, "whirld[remoteclip]==0.3.0", False),
        (
            '{"url":"file:///tmp/My%20Project","dir_info":{"editable":true}}',
            "/tmp/My Project[remoteclip]",
            True,
        ),
        (
            '{"url":"https://example.org/immutable.zip"}',
            "whirld[remoteclip] @ https://example.org/immutable.zip",
            False,
        ),
        (
            '{"url":"https://example.org/repo.git","vcs_info":{"vcs":"git","commit_id":"'
            + "a" * 40
            + '"}}',
            "whirld[remoteclip] @ git+https://example.org/repo.git@" + "a" * 40,
            False,
        ),
    ],
)
def test_repairs_preserve_installed_source_identity(
    monkeypatch: pytest.MonkeyPatch, origin: str | None, target: str, editable: bool
) -> None:
    """Unreleased installs retain their own source in a repair command."""
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "distribution",
        lambda name: PackageDistribution(origin),
    )
    assert DiagnosticsService._install_target("remoteclip") == (target, editable)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.org/archive.zip",
        "https://@example.org/archive.zip",
        "https://example.org/archive.zip?secret=token",
        "https://example.org/archive.zip#secret",
        "file://remote/private/path",
        "file:relative",
        "file:///tmp/%00secret",
        "http://example.org/archive.zip",
    ],
)
def test_source_repair_never_echoes_credentials_or_invalid_local_identity(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Ambiguous origins produce guide instructions without exposing source secrets."""
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "distribution",
        lambda name: PackageDistribution(json.dumps({"url": url})),
    )
    target, editable = DiagnosticsService._install_target("remoteclip")
    assert target is None and not editable
    assert "secret" not in DiagnosticsService._repair_runtime("remoteclip")


@pytest.mark.parametrize("pip_available", [False, True])
def test_repair_command_targets_current_interpreter_and_quotes_paths(
    monkeypatch: pytest.MonkeyPatch, pip_available: bool
) -> None:
    """Repair cannot accidentally install the runtime into a different Python."""
    executable = "/tmp/my env/.venv/bin/python"
    monkeypatch.setattr(doctor.sys, "executable", executable)
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        doctor.importlib.util,
        "find_spec",
        lambda name: object() if pip_available else None,
    )
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "distribution",
        lambda name: PackageDistribution(
            '{"url":"file:///tmp/Project%20Name","dir_info":{"editable":true}}'
        ),
    )
    command = shlex.split(
        DiagnosticsService._repair_runtime("remoteclip").splitlines()[0]
    )
    if pip_available:
        assert command[:4] == [executable, "-m", "pip", "install"]
    else:
        assert command[:5] == ["uv", "pip", "install", "--python", executable]
        assert command[5:7] == ["--torch-backend", "cpu"]
    assert command[-2:] == ["--editable", "/tmp/Project Name[remoteclip]"]


def test_inconsistent_report_readiness_is_rejected(
    tmp_path: Path, healthy_dependencies: None
) -> None:
    """Serialized reports cannot claim readiness while retaining blocking failures."""
    document = service(tmp_path).inspect().model_dump()
    document["ready_for_demo"] = False
    with pytest.raises(ValidationError, match="readiness"):
        DoctorReport.model_validate(document)


def test_installed_weights_remove_download_budget_from_full_report(
    tmp_path: Path, healthy_dependencies: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified existing cache requires only the bounded working-space budget."""
    payload = b"real hash fixture"
    target = service(tmp_path)
    entry = fake_entry(payload)
    install_fixture(target, entry, payload)
    monkeypatch.setattr(
        target,
        "_registry",
        lambda model: (
            entry,
            DiagnosticCheck(code="registry", status="pass", message="Fixture contract"),
        ),
    )
    report = target.inspect()
    assert report.ready_for_demo and report.model_installed
    assert report.model_download_bytes == 0
    assert report.disk_required_bytes == 64 * 1024**2
    assert check(report, "model").status == "pass"


@pytest.mark.parametrize("pip_available", [False, True])
def test_broken_runtime_repairs_explicitly_reinstall_using_packaged_constraints(
    monkeypatch: pytest.MonkeyPatch, pip_available: bool
) -> None:
    """A broken installed runtime needs reinstall, not an 'already satisfied' result."""
    monkeypatch.setattr(
        doctor.importlib.util,
        "find_spec",
        lambda name: object() if pip_available else None,
    )
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "distribution",
        lambda name: PackageDistribution(None),
    )
    command = shlex.split(
        DiagnosticsService._repair_runtime("remoteclip", reinstall=True)
    )
    assert ("--force-reinstall" if pip_available else "--reinstall") in command
    constraints = Path(command[command.index("--constraint") + 1])
    assert constraints.name == "requirements-quickstart.txt" and constraints.is_file()
    if pip_available:
        assert command[command.index("--extra-index-url") + 1] == (
            "https://download.pytorch.org/whl/cpu"
        )
    else:
        assert command[command.index("--torch-backend") + 1] == "cpu"


def test_packaged_quickstart_constraints_match_the_root_profile() -> None:
    """Wheel repair instructions match the installer profile maintained in source."""
    bundled = doctor.importlib.resources.files("whirld.onboarding").joinpath(
        "data/requirements-quickstart.txt"
    )
    root = Path(__file__).resolve().parents[3] / "requirements-quickstart.txt"
    assert bundled.read_bytes() == root.read_bytes()


def test_missing_runtime_rows_include_same_environment_repair(
    tmp_path: Path, healthy_dependencies: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed selected import blocks readiness and gives a concrete repair route."""

    def broken(distribution: str, module: str) -> DependencyCheck:
        """Keep healthy base imports while simulating an installed broken OpenCLIP."""
        return DependencyCheck(
            distribution=distribution,
            module=module,
            version="3.3.0",
            status="fail" if module == "open_clip" else "pass",
            message="OpenCLIP requires repair" if module == "open_clip" else "Ready",
        )

    monkeypatch.setattr(DiagnosticsService, "_dependency", staticmethod(broken))
    report = service(tmp_path).inspect()
    observation = check(report, "dependency:open_clip")
    assert not report.ready_for_demo and observation.status == "fail"
    assert observation.remedy is not None
    assert sys.executable in shlex.split(observation.remedy)
    assert "reinstall" in observation.remedy


@pytest.mark.parametrize(
    ("backend", "required", "excluded"),
    [
        ("clay-reference", "numpy", "torch"),
        ("clay", "einops", "terratorch"),
        ("remoteclip", "open_clip", "terratorch"),
        ("prithvi", "terratorch", "open_clip"),
    ],
)
def test_only_the_selected_backend_runtime_is_required(
    backend: str, required: str, excluded: str
) -> None:
    """Diagnostics avoid requiring every optional model family for a single workflow."""
    modules = {module for _, module in DiagnosticsService._requirements(backend)}
    assert required in modules and excluded not in modules
