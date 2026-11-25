"""Demo option boundaries, actual checkpoint checks and controlled failures."""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.demo.runner import DemoCpuRuntime, DemoOptions, DemoRunner
from whirld.errors import InvalidInputError


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2"])
def test_demo_threads_require_strict_positive_integers(
    tmp_path: Path, threads: object
) -> None:
    """Reject ambiguous settings before any diagnostics, download or output writes."""
    with pytest.raises(ValidationError):
        DemoOptions.model_validate({"output_dir": tmp_path, "threads": threads})


@pytest.fixture
def model_identity(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DemoRunner, Path, ModelEntry]:
    """Keep actual byte hashing while substituting a tiny curated fixture identity."""
    store = ModelStore()
    entry = store.registry.get("remoteclip-rn50")
    weights = b"independent checkpoint fixture"
    fields = entry.model_dump()
    fields["distribution"].update(
        sha256=hashlib.sha256(weights).hexdigest(), size_bytes=len(weights)
    )
    fixture = ModelEntry.model_validate(fields)
    manifest = Manifest(
        name=fixture.name,
        version=fixture.version,
        source_type="huggingface",
        sha256=fixture.distribution.sha256,
        weights_file="fixture.pt",
    )
    path = store.paths.model_file(fixture.name, manifest.weights_file)
    path.parent.mkdir(parents=True)
    path.write_bytes(weights)

    def get(registry: object, model: str) -> ModelEntry:
        """Return a controlled active registry contract for the tiny checkpoint."""
        return fixture

    def bundled(path: Path) -> ModelEntry:
        """Keep bundle/active contract comparison without creating a registry file."""
        return fixture

    monkeypatch.setattr(type(store.registry), "get", get)
    monkeypatch.setattr(ModelEntry, "from_registry_file", bundled)
    monkeypatch.setattr(ModelStore, "load_manifest", lambda store, name: manifest)
    return DemoRunner(DemoOptions(output_dir=tmp_path / "run")), path, fixture


def test_actual_checkpoint_sha_is_required_and_returned(
    model_identity: tuple[DemoRunner, Path, ModelEntry],
) -> None:
    """The report must identify actual verified bytes, not just a declared manifest."""
    demo, path, fixture = model_identity
    entry, actual = demo._verify_model()
    assert entry == fixture
    assert actual == hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_corruption_fails_before_inference(
    model_identity: tuple[DemoRunner, Path, ModelEntry],
) -> None:
    """Same-size changed weights cannot pass readiness or produce trusted provenance."""
    demo, path, fixture = model_identity
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(InvalidInputError, match="checksum"):
        demo._verify_model()


@pytest.mark.parametrize("mismatch", ["registry", "manifest"])
def test_changed_model_contract_is_rejected(
    model_identity: tuple[DemoRunner, Path, ModelEntry],
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    """Active overrides and stale installs cannot silently change the packaged demo."""
    demo, path, fixture = model_identity
    if mismatch == "registry":
        fields = fixture.model_dump()
        fields["description"] = "changed"
        changed = ModelEntry.model_validate(fields)
        monkeypatch.setattr(ModelEntry, "from_registry_file", lambda path: changed)
    else:
        monkeypatch.setattr(
            ModelStore,
            "load_manifest",
            lambda store, name: Manifest(
                name=fixture.name,
                version=fixture.version,
                source_type="huggingface",
                sha256="f" * 64,
                weights_file="fixture.pt",
            ),
        )
    with pytest.raises(InvalidInputError, match="contract"):
        demo._verify_model()


@pytest.mark.parametrize(
    "failure", [None, RuntimeError("failure"), KeyboardInterrupt()]
)
def test_cpu_scope_filters_only_expected_own_thread_initialization_notice(
    caplog: pytest.LogCaptureFixture, failure: BaseException | None
) -> None:
    """Checkpoint loading hides its temporary notice while unrelated warnings remain."""
    logger = logging.getLogger()
    before = tuple(logger.filters)
    caplog.set_level(logging.WARNING)
    expected = (
        "No pretrained weights loaded for model 'RN50'. Model initialized randomly."
    )

    def emit(
        message: str, path: str, thread: int, level: int = logging.WARNING
    ) -> None:
        """Send source-tagged log records through the same root route as OpenCLIP."""
        record = logging.LogRecord("root", level, path, 123, message, (), None)
        record.thread = thread
        logger.handle(record)

    try:
        with DemoCpuRuntime(2).activate():
            emit(expected, "/site-packages/open_clip/factory.py", threading.get_ident())
            emit(
                expected,
                "/site-packages/open_clip/factory.py",
                threading.get_ident() + 1,
            )
            emit(expected, "/other/factory.py", threading.get_ident())
            emit(
                expected.replace("RN50", "ViT-B-32"),
                "/open_clip/factory.py",
                threading.get_ident(),
            )
            emit("other warning", "/open_clip/factory.py", threading.get_ident())
            emit(
                expected, "/open_clip/factory.py", threading.get_ident(), logging.ERROR
            )
            if failure is not None:
                raise failure
    except BaseException as exc:
        assert exc is failure
    assert len(caplog.records) == 5
    assert caplog.records[-1].levelno == logging.ERROR
    assert tuple(logger.filters) == before
