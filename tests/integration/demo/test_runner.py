"""Offline end-to-end demo checks retain inference preparation and output writing."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import torch

import whirld
from tests.fixtures.demo import make_demo_resource
from whirld import api
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.demo import runner
from whirld.demo.runner import DemoOptions, DemoReport, DemoRunner
from whirld.errors import InvalidInputError
from whirld.models.base import InferenceContext, ModelBackend
from whirld.onboarding.doctor import DiagnosticCheck, DiagnosticsService, DoctorReport


class DemoBackend(ModelBackend):
    """Deterministic fake scores depending on actual normalized native RGB chips."""

    def __init__(self) -> None:
        """Record chip observations and bind a controlled failure switch."""
        super().__init__("remoteclip-rn50", "1.0.0", "cpu")
        self.chips: np.ndarray | None = None
        self.error: BaseException | None = None

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Reject accidental dispatch to the wrong model paradigm."""
        raise AssertionError("unexpected embedding")

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Consume real preprocessing and produce finite four-prompt probabilities."""
        self.chips = chips.copy()
        random.seed(111)
        np.random.seed(111)
        torch.default_generator.manual_seed(111)
        if self.error is not None:
            raise self.error
        means = chips.mean(axis=(1, 2, 3))
        logits = np.column_stack((means, -means, means * 0.5, np.zeros_like(means)))
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (values / values.sum(axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def demo_setup(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> DemoBackend:
    """Replace only network/optional-model seams around actual public classification."""
    resource = tmp_path / "resource"
    make_demo_resource(resource)
    monkeypatch.setattr("whirld.demo.sample.files", lambda package: resource)
    store = ModelStore()
    entry = store.registry.get("remoteclip-rn50")
    manifest = Manifest(
        name=entry.name,
        version=entry.version,
        source_type="huggingface",
        sha256=entry.distribution.sha256,
        weights_file="fixture.pt",
    )

    def installed(model_store: ModelStore, name: str) -> Manifest:
        """Provide registry-consistent installed metadata without optional weights."""
        assert name == entry.name
        return manifest

    def identity(demo: DemoRunner) -> tuple[ModelEntry, str]:
        """Keep the small pipeline test independent of the 408 MB checkpoint."""
        return entry, entry.distribution.sha256

    def inspect(service: DiagnosticsService, model: str) -> DoctorReport:
        """Represent a ready environment whose cached model is available."""
        return DoctorReport(
            model=model,
            python_version="3.13.7",
            executable="fixture-python",
            cache_path=whirld_home,
            model_installed=True,
            model_size_bytes=entry.distribution.size_bytes,
            model_download_bytes=0,
            disk_free_bytes=10**12,
            disk_required_bytes=0,
            dependencies=(),
            checks=(),
            ready_for_demo=True,
        )

    backend = DemoBackend()
    monkeypatch.setattr(ModelStore, "load_manifest", installed)
    monkeypatch.setattr(DemoRunner, "_verify_model", identity)
    monkeypatch.setattr(runner.DiagnosticsService, "inspect", inspect)
    monkeypatch.setattr(api, "load_backend", lambda *args, **kwargs: backend)
    return backend


def test_real_preprocessing_and_outputs_produce_viewable_first_result(
    tmp_path: Path, demo_setup: DemoBackend
) -> None:
    """Native chips retain valid zeros, scores, pixels and provenance."""
    output = tmp_path / "run"
    result = DemoRunner(DemoOptions(output_dir=output)).run()
    assert result.feature_count == 4 and result.model == "remoteclip-rn50"
    assert demo_setup.chips is not None
    assert demo_setup.chips.shape == (4, 3, 224, 224)
    # Valid DN zero is normalized; only source nodata takes the contract's zero fill.
    assert demo_setup.chips[0, 0, 0, 0] < -1
    assert demo_setup.chips[0, 0, 1, 1] == 0
    collection = json.loads(result.scores_path.read_text())
    assert len(collection["features"]) == 4
    assert collection["queries"] == list(runner.PROMPTS)
    html = result.preview_path.read_text()
    assert "data:image/png;base64," in html
    assert "land-cover confidence" not in html or "not calibrated" in html
    assert html.count("<polygon") == 4
    with rasterio.open(output / "preview.png") as image:
        assert image.shape == (448, 448) and image.count == 3
    assert (
        DemoReport.model_validate_json((output / "report.json").read_text()) == result
    )
    assert result.sample.units == "canonical_dn"


def test_existing_output_survives_without_readiness_or_acquisition(
    tmp_path: Path, demo_setup: DemoBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reruns preserve the complete earlier result before side effects."""
    output = tmp_path / "prior"
    output.mkdir()
    (output / "prior.txt").write_text("keep")

    def unexpected(*args: object, **kwargs: object) -> None:
        """Catch both diagnostic and model download calls during an invalid rerun."""
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr(runner.DiagnosticsService, "inspect", unexpected)
    monkeypatch.setattr(whirld, "pull", unexpected)
    with pytest.raises(InvalidInputError, match="exists"):
        DemoRunner(DemoOptions(output_dir=output)).run()
    assert (output / "prior.txt").read_text() == "keep"


@pytest.mark.parametrize("failure", [RuntimeError("failure"), KeyboardInterrupt()])
def test_inference_failure_removes_only_owned_run_and_restores_cpu_state(
    tmp_path: Path, demo_setup: DemoBackend, failure: BaseException
) -> None:
    """Errors preserve random/CPU state and neighboring output files."""
    before_threads = torch.get_num_threads()
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    demo_setup.error = failure
    output = tmp_path / "run"
    prior = tmp_path / "prior.txt"
    prior.write_text("keep")
    with pytest.raises(type(failure)):
        DemoRunner(DemoOptions(output_dir=output)).run()
    assert not output.exists()
    assert prior.read_text() == "keep"
    assert torch.get_num_threads() == before_threads
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    np.testing.assert_equal(after_numpy, before_numpy)


def test_preflight_failure_prevents_output_and_large_download(
    tmp_path: Path, demo_setup: DemoBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable runtimes fail actionably before checkpoint downloads."""
    original = runner.DiagnosticsService.inspect

    def blocked(service: DiagnosticsService, model: str) -> DoctorReport:
        """Keep the readiness report internally consistent while adding one blocker."""
        report = original(service, model)
        fields = report.model_dump()
        fields.update(
            ready_for_demo=False,
            checks=(
                DiagnosticCheck(
                    code="runtime", status="fail", message="missing", remedy="repair"
                ),
            ),
        )
        return DoctorReport.model_validate(fields)

    def unexpected(*args: object, **kwargs: object) -> None:
        """Catch unintended dependency/weight acquisition after failed checks."""
        raise AssertionError("unexpected download")

    monkeypatch.setattr(runner.DiagnosticsService, "inspect", blocked)
    monkeypatch.setattr(whirld, "pull", unexpected)
    with pytest.raises(InvalidInputError, match="missing repair"):
        DemoRunner(DemoOptions(output_dir=tmp_path / "run")).run()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("changed", ["model", "sensor", "device", "count", "schema"])
def test_wrong_output_contract_never_publishes_a_demo(
    tmp_path: Path,
    demo_setup: DemoBackend,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    """Reject changed result identity or shape before interpreting scores."""
    original = whirld.classify

    def classify(path: str | Path, **kwargs: Any) -> api.ClassifyResult:
        """Alter one returned artifact after real preprocessing and fake inference."""
        result = original(path, **kwargs)
        if changed == "model":
            result.feature_collection["model"] = "other-model"
        elif changed == "sensor":
            result.sensor = "other-sensor"
        elif changed == "device":
            result.device = "cuda"
        elif changed == "count":
            result.chips.pop()
        else:
            result.feature_collection = {"type": "FeatureCollection"}
        return result

    monkeypatch.setattr(whirld, "classify", classify)
    with pytest.raises(InvalidInputError):
        DemoRunner(DemoOptions(output_dir=tmp_path / "run")).run()
    assert not (tmp_path / "run").exists()


def test_missing_weights_are_pulled_after_validating_sample(
    tmp_path: Path, demo_setup: DemoBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified sample and successful preflight precede one ordinary pull."""
    original = runner.DiagnosticsService.inspect
    downloads: list[str] = []

    def missing(service: DiagnosticsService, model: str) -> DoctorReport:
        """Keep missing weights as a nonblocking readiness state."""
        return original(service, model).model_copy(
            update={"model_installed": False, "model_download_bytes": 408397807}
        )

    def pull(model: str) -> None:
        """Assert the small input already exists when the first large fetch begins."""
        assert (tmp_path / "run/starter-rgb.tif").is_file()
        downloads.append(model)

    monkeypatch.setattr(runner.DiagnosticsService, "inspect", missing)
    monkeypatch.setattr(whirld, "pull", pull)
    DemoRunner(DemoOptions(output_dir=tmp_path / "run")).run()
    assert downloads == ["remoteclip-rn50"]


def test_bad_bundled_sample_prevents_weight_download(
    tmp_path: Path, demo_setup: DemoBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged wheel should fail cheaply before acquiring hundreds of megabytes."""
    image = tmp_path / "resource/data/starter-rgb.tif"
    image.write_bytes(b"bad")
    original = runner.DiagnosticsService.inspect
    monkeypatch.setattr(
        runner.DiagnosticsService,
        "inspect",
        lambda service, model: original(service, model).model_copy(
            update={"model_installed": False}
        ),
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        """Catch premature checkpoint acquisition from a broken package."""
        raise AssertionError("unexpected download")

    monkeypatch.setattr(whirld, "pull", unexpected)
    with pytest.raises(InvalidInputError, match="checksum"):
        DemoRunner(DemoOptions(output_dir=tmp_path / "run")).run()
    assert not (tmp_path / "run").exists()


def test_final_report_failure_removes_completed_artifacts(
    tmp_path: Path, demo_setup: DemoBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Success requires provenance to be persisted with the completed preview."""
    original = Path.write_text

    def write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        """Simulate a full disk at final report publication after the preview exists."""
        if path.name == "report.json":
            assert (path.parent / "preview.html").is_file()
            raise OSError("full disk")
        return original(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match="full disk"):
        DemoRunner(DemoOptions(output_dir=tmp_path / "run")).run()
    assert not (tmp_path / "run").exists()
