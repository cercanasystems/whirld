"""Weight acquisition, checksum verification, and local manifests.

``whirld pull`` resolves a registry entry, acquires the model artifact, verifies
its sha256, and writes a local ``manifest.json`` (PRD section 5.2). This module
implements that contract for two source types behind one interface:

* ``reference`` — the offline backend. Whirld *materializes* a tiny, deterministic
  weights blob locally (derived from the model's seed and embedding dimension)
  instead of downloading hundreds of megabytes. The full resolve -> materialize ->
  sha256 verify -> manifest path runs unchanged and entirely offline.
* ``huggingface`` — the real-weights path. Wired here (lazy ``huggingface_hub``
  import) but not exercised in the walking-skeleton build.

The deterministic blob is the canonical JSON document::

    {"backend":"reference","embed_dim":<N>,"model":"<name>","seed":<S>,"version":"<V>"}

serialized with sorted keys and no whitespace. Its sha256 is what the registry
entry declares, so checksum verification is meaningful even offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .. import config
from ..errors import (
    ChecksumMismatchError,
    ModelNotInstalledError,
    NetworkError,
    SecurityError,
    WhirldError,
)
from ..logging_setup import get_logger
from . import governance
from .registry import ModelEntry, Registry

_log = get_logger("core.fetch")


def _enforce_weights_security(entry: ModelEntry) -> None:
    """Refuse to acquire a community model that ships executable (pickle) weights.

    Pickle weights (``.pt``/``.ckpt``/…) run arbitrary code on load. Only
    first-party, maintainer-curated entries may use pickle; community entries must
    ship ``safetensors`` (PRD §8 governance).

    Args:
        entry: The registry entry being pulled.

    Raises:
        SecurityError: The entry is ``community`` and its weights are pickle.
    """
    if entry.trust != governance.FIRST_PARTY and governance.weights_are_pickle(entry):
        primary = entry.source.files[0] if entry.source.files else "weights"
        raise SecurityError(
            f"Refusing to pull '{entry.name}': it is a community entry whose "
            f"weights ('{primary}') are a pickle format.\n"
            f"       Pickle deserialization executes arbitrary code on load. "
            f"Community models must ship safetensors;\n"
            f"       only first-party curated entries may use pickle."
        )


_REFERENCE_WEIGHTS_FILENAME = "weights.ref.json"
_CHUNK = 1024 * 1024


class Manifest(BaseModel):
    """Local record written to ``<model_dir>/manifest.json`` after a pull.

    Attributes:
        name: Model identifier.
        version: Model version pulled.
        source_type: ``reference`` or ``huggingface``.
        sha256: Verified checksum of the weights artifact.
        weights_file: Filename of the weights artifact within the model dir.
        embed_dim: Embedding dimensionality (embedding models), else ``None``.
        seed: Deterministic seed for the reference backend, else ``None``.
        quantized: Whether a quantized variant was pulled.
        downloaded_at: UTC ISO-8601 timestamp of the pull.
    """

    name: str
    version: str
    source_type: str
    sha256: str
    weights_file: str
    embed_dim: int | None = None
    seed: int | None = None
    quantized: bool = False
    downloaded_at: str = Field(default="")


def sha256_file(path: Path) -> str:
    """Compute the sha256 hex digest of a file, streaming in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_blob_bytes(name: str, version: str, embed_dim: int, seed: int) -> bytes:
    """Return the canonical deterministic reference weights blob.

    Args:
        name: Model identifier.
        version: Model version.
        embed_dim: Embedding dimensionality.
        seed: Deterministic backend seed.

    Returns:
        The canonical JSON bytes (sorted keys, no whitespace).
    """
    payload = {
        "backend": "reference",
        "model": name,
        "version": version,
        "embed_dim": embed_dim,
        "seed": seed,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def is_installed(name: str, paths: config.WhirldPaths | None = None) -> bool:
    """Return whether a model has a manifest in the local cache.

    Args:
        name: Model identifier.
        paths: Path bundle; defaults to the current environment.
    """
    paths = paths or config.get_paths()
    return paths.model_manifest(name).exists()


def remove_model(name: str, paths: config.WhirldPaths | None = None) -> None:
    """Delete a cached model's directory (weights + manifest).

    Args:
        name: Model identifier.
        paths: Path bundle; defaults to the current environment.

    Raises:
        ModelNotInstalledError: The model is not installed.
    """
    import shutil

    paths = paths or config.get_paths()
    model_dir = paths.model_dir(name)
    if not model_dir.exists():
        raise ModelNotInstalledError(
            f"Model '{name}' is not installed; nothing to remove."
        )
    shutil.rmtree(model_dir)
    _log.info("Removed model '%s' from %s", name, model_dir)


def remove_all(paths: config.WhirldPaths | None = None) -> list[str]:
    """Delete all cached models, preserving the registry (PRD section 11).

    Args:
        paths: Path bundle; defaults to the current environment.

    Returns:
        The sorted names of the models that were removed.
    """
    import shutil

    paths = paths or config.get_paths()
    if not paths.models_dir.exists():
        return []
    removed = sorted(d.name for d in paths.models_dir.iterdir() if d.is_dir())
    for name in removed:
        shutil.rmtree(paths.model_dir(name))
    _log.info("Removed %d model(s): %s", len(removed), ", ".join(removed) or "(none)")
    return removed


def load_manifest(name: str, paths: config.WhirldPaths | None = None) -> Manifest:
    """Load the local manifest for an installed model.

    Args:
        name: Model identifier.
        paths: Path bundle; defaults to the current environment.

    Returns:
        The parsed :class:`Manifest`.

    Raises:
        ModelNotInstalledError: The model has not been pulled.
    """
    paths = paths or config.get_paths()
    manifest_path = paths.model_manifest(name)
    if not manifest_path.exists():
        raise ModelNotInstalledError(
            f"Model '{name}' is not installed.\n"
            f"       Pull it first:  whirld pull {name}"
        )
    return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def pull(
    name: str,
    *,
    force: bool = False,
    quantize: str | None = None,
    paths: config.WhirldPaths | None = None,
    registry: Registry | None = None,
) -> Manifest:
    """Download/materialize, verify, and cache a model.

    Args:
        name: Model identifier to pull.
        force: Re-acquire even if already cached.
        quantize: Quantized variant to pull (e.g. ``int8``); deferred — raises
            if requested in this build.
        paths: Path bundle; defaults to the current environment.
        registry: Registry instance; defaults to a freshly seeded one.

    Returns:
        The written :class:`Manifest`.

    Raises:
        ModelNotFoundError: Model is not in the registry.
        ChecksumMismatchError: Verified checksum does not match the registry.
        NetworkError: A network acquisition failed (huggingface sources).
        SecurityError: A community entry ships pickle (not safetensors) weights.
        WhirldError: Quantization requested (deferred) or unknown source type.
    """
    paths = paths or config.get_paths()
    registry = registry or Registry(paths)
    paths.ensure_dirs()

    if quantize is not None:
        raise WhirldError(
            "Quantized variants are not available in this build.\n"
            "       Pull the full-precision model instead:  "
            f"whirld pull {name}"
        )

    entry = registry.get(name)
    _enforce_weights_security(entry)

    if is_installed(name, paths) and not force:
        _log.info("Model '%s' already cached; use --force to re-pull.", name)
        return load_manifest(name, paths)

    model_dir = paths.model_dir(name)
    model_dir.mkdir(parents=True, exist_ok=True)

    if entry.source.type == "reference":
        manifest = _pull_reference(entry, model_dir)
    elif entry.source.type == "huggingface":
        manifest = _pull_huggingface(entry, model_dir)
    else:
        raise WhirldError(
            f"Unknown source type '{entry.source.type}' for model '{name}'."
        )

    paths.model_manifest(name).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    _log.info("Saved %s to %s", name, model_dir)
    return manifest


def _pull_reference(entry: ModelEntry, model_dir: Path) -> Manifest:
    """Materialize and verify the deterministic reference weights blob.

    Args:
        entry: Registry entry being pulled.
        model_dir: Destination directory for the model.

    Returns:
        The manifest describing the materialized artifact.

    Raises:
        ChecksumMismatchError: The blob's sha256 does not match the registry.
        WhirldError: The reference source is missing a seed or embed_dim.
    """
    seed = entry.source.seed
    embed_dim = entry.output.embed_dim
    if seed is None or embed_dim is None:
        raise WhirldError(
            f"Reference source for '{entry.name}' must declare source.seed and "
            f"output.embed_dim."
        )

    blob = reference_blob_bytes(entry.name, entry.version, embed_dim, seed)
    weights_path = model_dir / _REFERENCE_WEIGHTS_FILENAME
    weights_path.write_bytes(blob)

    actual = sha256_file(weights_path)
    if actual != entry.distribution.sha256:
        weights_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"Checksum mismatch for '{entry.name}'.\n"
            f"       expected sha256: {entry.distribution.sha256}\n"
            f"       actual sha256:   {actual}\n"
            f"       The cached file was deleted. Re-run 'whirld pull "
            f"{entry.name}'."
        )

    return Manifest(
        name=entry.name,
        version=entry.version,
        source_type="reference",
        sha256=actual,
        weights_file=_REFERENCE_WEIGHTS_FILENAME,
        embed_dim=embed_dim,
        seed=seed,
        quantized=False,
        downloaded_at=_utc_now_iso(),
    )


def _pull_huggingface(entry: ModelEntry, model_dir: Path) -> Manifest:
    """Download real weights from Hugging Face Hub.

    Lazily imports ``huggingface_hub`` so the dependency (``hf`` extra) is
    optional. Exercised by real models such as ``clay-v1.5``.

    Args:
        entry: Registry entry being pulled.
        model_dir: Destination directory for the model.

    Returns:
        The manifest describing the downloaded artifact.

    Raises:
        NetworkError: The optional dependency is missing or the download failed.
        ChecksumMismatchError: The downloaded file's checksum is wrong.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise NetworkError(
            "Real Hugging Face weights require the optional 'hf' extra.\n"
            "       Install it:  pip install 'whirld[hf]'"
        ) from exc

    if not entry.source.repo or not entry.source.files:
        raise NetworkError(
            f"Hugging Face source for '{entry.name}' is missing repo/files."
        )

    # Download every declared file (e.g. weights + an auxiliary config); the first
    # file is the primary artifact whose sha256 is verified.
    try:
        downloaded_paths = [
            Path(
                hf_hub_download(
                    repo_id=entry.source.repo,
                    filename=filename,
                    revision=entry.source.revision,
                    local_dir=str(model_dir),
                )
            )
            for filename in entry.source.files
        ]
    except Exception as exc:  # network/hub failures
        raise NetworkError(
            f"Failed to download '{entry.name}' from Hugging Face Hub.\n"
            f"       {exc}"
        ) from exc

    downloaded = downloaded_paths[0]
    actual = sha256_file(downloaded)
    if actual != entry.distribution.sha256:
        downloaded.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"Checksum mismatch for '{entry.name}' (expected "
            f"{entry.distribution.sha256}, got {actual}). File deleted."
        )

    # Record the path relative to the model dir; HF may nest it (e.g. v1.5/...).
    try:
        weights_file = str(downloaded.relative_to(model_dir))
    except ValueError:
        weights_file = downloaded.name

    return Manifest(
        name=entry.name,
        version=entry.version,
        source_type="huggingface",
        sha256=actual,
        weights_file=weights_file,
        embed_dim=entry.output.embed_dim,
        seed=entry.source.seed,
        quantized=False,
        downloaded_at=_utc_now_iso(),
    )


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``Z``-suffixed string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
