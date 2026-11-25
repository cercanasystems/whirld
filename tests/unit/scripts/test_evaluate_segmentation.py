"""Evaluation command defaults, fixed protocol and actionable failure handling."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import evaluate_segmentation as command
from whirld.errors import ModelNotInstalledError
from whirld.evaluation.errors import EvaluationError
from whirld.evaluation.runner import SuiteLoader
from whirld.evaluation.schemas import (
    EvaluationOptions,
    EvaluationReport,
    ModelIdentity,
    RuntimeIdentity,
    SuiteDocument,
)


class CommandProbe:
    """Observe CLI-to-schema boundaries while replacing only execution and loading."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bind a valid real suite and typed fake report without weights/network."""
        self.document = SuiteLoader.load()
        self.options: list[EvaluationOptions] = []
        self.manifests: list[Path | None] = []
        self.failure: BaseException | None = None
        probe = self

        def load(path: Path | None = None) -> SuiteDocument:
            """Observe explicit/default manifest selection and return typed data."""
            probe.manifests.append(path)
            return probe.document

        def configure() -> None:
            """Leave the test's capture handlers unchanged during CLI execution."""

        class ReportRunner:
            """Return typed command output instead of performing model inference."""

            def __init__(
                self, document: SuiteDocument, options: EvaluationOptions
            ) -> None:
                """Retain schema-validated arguments exactly as the CLI passes them."""
                assert document == probe.document
                self.options = options
                probe.options.append(options)

            def run(self) -> EvaluationReport:
                """Return a minimal valid report or raise a requested runtime error."""
                if probe.failure is not None:
                    raise probe.failure
                suite = probe.document.suite
                return EvaluationReport(
                    suite=suite,
                    manifest_sha256=probe.document.sha256,
                    model=ModelIdentity(
                        name=suite.model,
                        version="2.0.0",
                        revision=suite.model_revision,
                        checkpoint_sha256=suite.model_sha256,
                        config_sha256=suite.model_config_sha256,
                        registry_sha256="0" * 64,
                    ),
                    runtime=RuntimeIdentity(
                        python="3.13.7",
                        platform="fixture",
                        whirld="0.3.0",
                        numpy="2.5.0",
                        rasterio="1.5.2",
                        torch="2.12.1",
                        terratorch="1.2.8",
                    ),
                    options=self.options,
                    created_at=datetime.now(UTC),
                    cases=(),
                    splits=(),
                )

        monkeypatch.setattr(command.SuiteLoader, "load", load)
        monkeypatch.setattr(command, "EvaluationRunner", ReportRunner)
        monkeypatch.setattr(command, "configure_logging", configure)


def test_default_command_preserves_validation_only_fixed_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The convenience command does not silently score test cases or tune threshold."""
    probe = CommandProbe(monkeypatch)
    with caplog.at_level(logging.INFO):
        result = command.EvaluationCommand().run(
            ["--output-dir", str(tmp_path / "out")]
        )
    assert result == 0 and probe.manifests == [None]
    assert len(probe.options) == 1
    options = probe.options[0]
    assert options.output_dir == tmp_path / "out"
    assert options.cache_dir == Path("evaluation-runs/cache")
    assert options.split == "validation" and options.threshold == 0.5
    assert options.batch_size == 1 and options.cache_mb == 8 and options.threads == 2
    assert not options.offline and not options.output_dir.exists()
    assert str((options.output_dir / "report.json").resolve()) in caplog.text


def test_command_passes_explicit_offline_and_split_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths and bounded execution settings cross the command boundary unchanged."""
    probe = CommandProbe(monkeypatch)
    manifest = tmp_path / "suite.json"
    result = command.EvaluationCommand().run(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--manifest",
            str(manifest),
            "--split",
            "all",
            "--offline",
            "--batch-size",
            "2",
            "--cache-mb",
            "16",
            "--threads",
            "1",
        ]
    )
    assert result == 0 and probe.manifests == [manifest]
    assert probe.options[0].split == "all" and probe.options[0].offline
    assert probe.options[0].cache_dir == tmp_path / "cache"
    assert probe.options[0].batch_size == 2
    assert probe.options[0].cache_mb == 16 and probe.options[0].threads == 1


@pytest.mark.parametrize("option", ["--batch-size", "--cache-mb", "--threads"])
def test_nonpositive_execution_settings_fail_before_suite_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    option: str,
) -> None:
    """Invalid bounded-execution settings cannot trigger data or model operations."""
    probe = CommandProbe(monkeypatch)
    result = command.EvaluationCommand().run(
        ["--output-dir", str(tmp_path / "out"), option, "0"]
    )
    assert result == 1 and "Evaluation failed" in caplog.text
    assert probe.manifests == [] and probe.options == []


@pytest.mark.parametrize(
    "failure,exit_code",
    [
        (EvaluationError("checksum differs"), 7),
        (ModelNotInstalledError("pull the model first"), 3),
        (FileExistsError("previous output exists"), 1),
        (OSError("disk unavailable"), 1),
    ],
)
def test_command_preserves_domain_exit_codes_and_local_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: BaseException,
    exit_code: int,
) -> None:
    """Runtime errors remain actionable without an unhandled CLI traceback."""
    probe = CommandProbe(monkeypatch)
    probe.failure = failure
    assert (
        command.EvaluationCommand().run(["--output-dir", str(tmp_path / "out")])
        == exit_code
    )
    assert str(failure) in caplog.text


def test_command_does_not_swallow_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interruption propagates after the runner's resource cleanup contract."""
    probe = CommandProbe(monkeypatch)
    probe.failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        command.EvaluationCommand().run(["--output-dir", str(tmp_path / "out")])


def test_help_is_available_without_a_model_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The help surface exposes split/offline settings and no threshold tuning flag."""
    probe = CommandProbe(monkeypatch)
    with pytest.raises(SystemExit) as result:
        command.EvaluationCommand().run(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--split" in output and "--offline" in output
    assert "--threshold" not in output
    assert probe.manifests == [] and probe.options == []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--output-dir", "unused", "--threshold", "0.7"],
        ["--output-dir", "unused", "--split", "train"],
    ],
)
def test_argparse_rejects_missing_output_or_protocol_changes(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    """Malformed commands fail before even loading the frozen suite."""
    probe = CommandProbe(monkeypatch)
    with pytest.raises(SystemExit) as result:
        command.EvaluationCommand().run(arguments)
    assert result.value.code == 2
    assert probe.manifests == [] and probe.options == []
