"""Read-only environment and model diagnostics for an installed Whirld."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import os
import platform
import re
import shlex
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .. import config
from ..core.fetch import Manifest, sha256_file
from ..core.registry import ModelEntry
from ..errors import ModelNotFoundError, WhirldError


class DoctorError(WhirldError):
    """Setup checks found a condition preventing a reliable first demo."""

    exit_code = 7


class DiagnosticCheck(BaseModel):
    """One actionable setup observation; warnings do not block the demo."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    code: str
    status: Literal["pass", "warn", "fail"]
    message: str
    remedy: str | None = None


class DependencyCheck(BaseModel):
    """An installed distribution and the result of importing its runtime module."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    distribution: str
    module: str
    version: str | None
    status: Literal["pass", "fail"]
    message: str
    remedy: str | None = None


class DoctorReport(BaseModel):
    """Typed setup snapshot, including missing-weight acquisition requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str
    python_version: str
    executable: str
    device: Literal["cpu"] = "cpu"
    cache_path: Path
    model_installed: bool
    model_size_bytes: int = Field(ge=0)
    model_download_bytes: int = Field(ge=0)
    disk_free_bytes: int | None = Field(ge=0)
    disk_required_bytes: int = Field(ge=0)
    dependencies: tuple[DependencyCheck, ...]
    checks: tuple[DiagnosticCheck, ...]
    ready_for_demo: bool

    @model_validator(mode="after")
    def consistent_readiness(self) -> DoctorReport:
        """Reject reports whose readiness contradicts their failure observations."""
        if self.ready_for_demo != all(check.status != "fail" for check in self.checks):
            raise ValueError("readiness must reflect every diagnostic check")
        return self

    def require_ready(self) -> None:
        """Raise DoctorError when setup blockers must be repaired before inference."""
        if not self.ready_for_demo:
            failures = "; ".join(
                check.message for check in self.checks if check.status == "fail"
            )
            raise DoctorError(
                f"Cannot run the demo: {failures} "
                f"Run whirld doctor --model {self.model} for repair commands."
            )


class _OriginDirectory(BaseModel):
    """Relevant editable-origin metadata at the installed-package boundary."""

    editable: bool = False


class _PackageOrigin(BaseModel):
    """PEP 610 source fields used to preserve the actual installation source."""

    url: str
    dir_info: _OriginDirectory | None = None
    vcs_info: _VersionControlOrigin | None = None


class _VersionControlOrigin(BaseModel):
    """Immutable git origin metadata, when the installation came from a checkout."""

    vcs: str
    commit_id: str


class DiagnosticsService:
    """Inspect setup without downloading, seeding a registry or changing cache files.

    Dependency imports happen only when inspect is called. Filesystem permissions
    are estimated from the nearest existing directory, rather than tested by
    writing. Model verification is a read-only snapshot; inference still uses the
    ModelStore installation lease to protect the actual backend load.
    """

    def __init__(self, paths: config.WhirldPaths | None = None) -> None:
        """Bind explicit or environment-selected cache paths without creating them."""
        self._paths = paths or config.get_paths()

    @staticmethod
    def _requirements(backend: str) -> tuple[tuple[str, str], ...]:
        """Return distribution/module names needed by the selected local backend."""
        common = (("numpy", "numpy"), ("rasterio", "rasterio"))
        if backend == "clay-reference":
            return common
        tensor = (("torch", "torch"), ("huggingface_hub", "huggingface_hub"))
        if backend == "clay":
            return (
                common + tensor + (("einops", "einops"), ("safetensors", "safetensors"))
            )
        if backend == "remoteclip":
            return (
                common
                + tensor
                + (
                    ("open_clip_torch", "open_clip"),
                    ("safetensors", "safetensors"),
                )
            )
        return (
            common + tensor + (("terratorch", "terratorch"), ("torchgeo", "torchgeo"))
        )

    @staticmethod
    def _extra(backend: str) -> str | None:
        """Map the installed backend contract to the existing optional extra."""
        return {
            "clay": "hf",
            "remoteclip": "remoteclip",
            "prithvi": "prithvi",
        }.get(backend)

    @staticmethod
    def _install_target(extra: str | None) -> tuple[str | None, bool]:
        """Resolve a safe installed origin, returning target and editable status.

        Credentials, query strings, fragments and unsupported URL schemes are not
        reflected in repair output. Missing PEP 610 metadata uses the installed
        index version; malformed or credentialed origins require guide instructions.
        """
        try:
            distribution = importlib.metadata.distribution("whirld")
            raw = distribution.read_text("direct_url.json")
            package = f"whirld[{extra}]" if extra else "whirld"
            if raw is None:
                return f"{package}=={distribution.version}", False
            origin = _PackageOrigin.model_validate_json(raw)
            parsed = urlsplit(origin.url)
            if (
                parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or any(ord(character) < 32 for character in origin.url)
            ):
                return None, False
            if parsed.scheme == "file":
                if parsed.netloc.lower() not in {"", "localhost"}:
                    return None, False
                path = Path(unquote(parsed.path))
                if not path.is_absolute() or any(
                    ord(character) < 32 for character in str(path)
                ):
                    return None, False
                if origin.dir_info is not None and origin.dir_info.editable:
                    return str(path) + (f"[{extra}]" if extra else ""), True
            elif parsed.scheme != "https" or not parsed.hostname:
                return None, False
            if origin.vcs_info is not None:
                if origin.vcs_info.vcs != "git" or not re.fullmatch(
                    r"[a-fA-F0-9]{40}", origin.vcs_info.commit_id
                ):
                    return None, False
                return (
                    f"{package} @ git+{origin.url}@{origin.vcs_info.commit_id}",
                    False,
                )
            return f"{package} @ {origin.url}", False
        except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
            return None, False

    @classmethod
    def _repair_runtime(cls, backend: str, *, reinstall: bool = False) -> str:
        """Give a same-interpreter installation command without assuming PyPI upload."""
        target, editable = cls._install_target(cls._extra(backend))
        if target is None:
            return (
                "Follow the tested installation in the quickstart guide for this "
                f"interpreter: {sys.executable}"
            )
        try:
            pip_available = importlib.util.find_spec("pip") is not None
        except (ImportError, ValueError):
            pip_available = False
        if pip_available:
            command = [sys.executable, "-m", "pip", "install"]
            if reinstall:
                command.append("--force-reinstall")
            if platform.system() == "Linux" and backend == "remoteclip":
                command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
        else:
            command = ["uv", "pip", "install", "--python", sys.executable]
            if reinstall:
                command.append("--reinstall")
            if platform.system() == "Linux" and backend == "remoteclip":
                command += ["--torch-backend", "cpu"]
        if backend == "remoteclip":
            constraints = importlib.resources.files("whirld.onboarding").joinpath(
                "data/requirements-quickstart.txt"
            )
            command += ["--constraint", str(constraints)]
        if editable:
            command.append("--editable")
        command.append(target)
        return shlex.join(command)

    @staticmethod
    def _dependency(distribution: str, module: str) -> DependencyCheck:
        """Read a distribution version and import its selected runtime module."""
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return DependencyCheck(
                distribution=distribution,
                module=module,
                version=None,
                status="fail",
                message=f"{distribution} is not installed in this interpreter.",
            )
        try:
            importlib.import_module(module)
        except Exception as exc:
            if (
                platform.system() == "Linux"
                and isinstance(exc, (ImportError, OSError))
                and "libexpat.so" in str(exc).lower()
            ):
                return DependencyCheck(
                    distribution=distribution,
                    module=module,
                    version=version,
                    status="fail",
                    message=(
                        f"{distribution} needs the missing operating-system "
                        "Expat XML runtime (libexpat.so)."
                    ),
                    remedy=(
                        "Install your Linux distribution's Expat runtime. "
                        "On Debian/Ubuntu, run with administrator privileges: "
                        "apt-get update && apt-get install libexpat1. "
                        "Then rerun whirld doctor."
                    ),
                )
            return DependencyCheck(
                distribution=distribution,
                module=module,
                version=version,
                status="fail",
                message=f"{distribution} cannot be imported ({type(exc).__name__}).",
            )
        return DependencyCheck(
            distribution=distribution,
            module=module,
            version=version,
            status="pass",
            message=f"{distribution} {version} imports successfully.",
        )

    def _registry(self, model: str) -> tuple[ModelEntry, DiagnosticCheck]:
        """Read the bundled contract and diagnose an existing conflicting override."""
        config.validate_model_name(model)
        bundled = config.bundled_registry_dir() / "models" / f"{model}.yaml"
        if not bundled.is_file():
            raise ModelNotFoundError(f"No bundled model named '{model}'.")
        entry = ModelEntry.from_registry_file(bundled)
        active = self._paths.registry_models_dir / f"{model}.yaml"
        try:
            active = config.cache_path(self._paths.registry_models_dir, f"{model}.yaml")
            if active.exists():
                current = ModelEntry.from_registry_file(active)
                if current != entry:
                    raise DoctorError(
                        "Active model registry differs from this installation."
                    )
        except (WhirldError, OSError):
            return entry, DiagnosticCheck(
                code="registry",
                status="fail",
                message="Active model registry differs from the bundled contract.",
                remedy=(
                    f"Review {active}; restore the bundled entry or choose a fresh "
                    "WHIRLD_HOME, then rerun doctor."
                ),
            )
        return entry, DiagnosticCheck(
            code="registry",
            status="pass",
            message="Bundled model contract is available.",
        )

    @staticmethod
    def _existing_directory(path: Path) -> Path:
        """Find a real existing directory that would contain a future cache root."""
        current = path
        while not current.exists() and current != current.parent:
            current = current.parent
        return current

    def _model(self, entry: ModelEntry) -> tuple[bool, DiagnosticCheck]:
        """Validate a manifest and its required files without cache mutation."""
        path = self._paths.model_manifest(entry.name)
        executable = str(
            Path(sysconfig.get_path("scripts"))
            / ("whirld.exe" if os.name == "nt" else "whirld")
        )
        repair = shlex.join([executable, "pull", entry.name, "--force"])
        try:
            present = path.exists()
        except OSError:
            return False, DiagnosticCheck(
                code="model",
                status="fail",
                message="Installed model manifest cannot be inspected.",
                remedy=f"Check access to {self._paths.home}, then rerun doctor.",
            )
        if not present:
            return False, DiagnosticCheck(
                code="model",
                status="warn",
                message="Model weights need verified acquisition before inference.",
            )
        try:
            manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
            if not manifest.matches_entry(entry):
                raise DoctorError(
                    "Installed manifest differs from the current contract."
                )
            required = set(manifest.required_files) | {manifest.weights_file}
            if not manifest.required_files and entry.source.type == "huggingface":
                required.update(entry.source.files)
            for filename in required:
                artifact = self._paths.model_file(entry.name, filename)
                expected = manifest.artifact_sizes.get(filename)
                if not artifact.is_file() or (
                    expected is not None and artifact.stat().st_size != expected
                ):
                    raise DoctorError(
                        "Required model artifacts are missing or truncated."
                    )
            weights = self._paths.model_file(entry.name, manifest.weights_file)
            if (
                weights.stat().st_size != entry.distribution.size_bytes
                or sha256_file(weights) != entry.distribution.sha256
            ):
                raise DoctorError("Cached checkpoint differs from its pinned SHA-256.")
        except (WhirldError, ValidationError, OSError, ValueError) as exc:
            return False, DiagnosticCheck(
                code="model",
                status="fail",
                message=(
                    exc.message
                    if isinstance(exc, DoctorError)
                    else f"Installed model is invalid ({type(exc).__name__})."
                ),
                remedy=repair,
            )
        return True, DiagnosticCheck(
            code="model", status="pass", message="Cached model SHA-256 is verified."
        )

    def inspect(self, model: str = "remoteclip-rn50") -> DoctorReport:
        """Inspect the selected model's CPU setup without automatic environment repair.

        Args:
            model: Bundled model identifier; RN50 is the first-run default.

        Returns:
            A report with actionable blockers, nonblocking missing weights and
            conservative free-space requirements. No weights are loaded and no
            network calls are made.

        Raises:
            WhirldError: The model identifier or cache path violates registry or
                filesystem security rules. Ordinary setup failures are report rows.
            KeyboardInterrupt: An interrupted import/hash propagates normally.
        """
        entry, registry = self._registry(model)
        backend = entry.backend
        if backend is None:
            raise DoctorError("The bundled model must declare a runtime backend.")
        version = tuple(sys.version_info[:3])
        checks = [registry]
        checks.append(
            DiagnosticCheck(
                code="python",
                status=(
                    "fail"
                    if version < (3, 13, 7)
                    else "pass" if version == (3, 13, 7) else "warn"
                ),
                message=(
                    "Python "
                    + ".".join(map(str, version))
                    + "; tested with Python 3.13.7."
                ),
                remedy=(
                    "uv venv --python 3.13.7 .venv" if version < (3, 13, 7) else None
                ),
            )
        )
        dependencies = tuple(
            self._dependency(distribution, module)
            for distribution, module in self._requirements(backend)
        )
        checks.extend(
            DiagnosticCheck(
                code=f"dependency:{dependency.module}",
                status=dependency.status,
                message=dependency.message,
                remedy=(
                    dependency.remedy
                    or self._repair_runtime(
                        backend, reinstall=dependency.version is not None
                    )
                    if dependency.status == "fail"
                    else None
                ),
            )
            for dependency in dependencies
        )
        installed, model_check = self._model(entry)
        checks.append(model_check)
        try:
            ancestor = self._existing_directory(self._paths.home)
            writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
        except OSError:
            ancestor = self._paths.home
            writable = False
        checks.append(
            DiagnosticCheck(
                code="cache",
                status="pass" if writable else "fail",
                message=(
                    f"Cache permission estimate: {self._paths.home} "
                    + ("can be written." if writable else "cannot be written.")
                ),
                remedy=None if writable else "Set WHIRLD_HOME to a writable directory.",
            )
        )
        download = 0 if installed else entry.distribution.size_bytes
        disk_required = 2 * download + 64 * 1024**2
        try:
            free = shutil.disk_usage(ancestor).free
        except OSError:
            free = None
        checks.append(
            DiagnosticCheck(
                code="disk",
                status="pass" if free is not None and free >= disk_required else "fail",
                message=f"Free cache bytes: {free}; budget: {disk_required}.",
                remedy=(
                    None
                    if free is not None and free >= disk_required
                    else "Free disk space or set WHIRLD_HOME to a filesystem with room."
                ),
            )
        )
        checks.append(
            DiagnosticCheck(
                code="device",
                status="pass",
                message="CPU selected; a GPU is not required.",
            )
        )
        return DoctorReport(
            model=entry.name,
            python_version=".".join(map(str, version)),
            executable=sys.executable,
            cache_path=self._paths.home,
            model_installed=installed,
            model_size_bytes=entry.distribution.size_bytes,
            model_download_bytes=download,
            disk_free_bytes=free,
            disk_required_bytes=disk_required,
            dependencies=dependencies,
            checks=tuple(checks),
            ready_for_demo=all(check.status != "fail" for check in checks),
        )
