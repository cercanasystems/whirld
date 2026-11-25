"""Installed command boundaries retain useful preflight and first-result behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle
from typer.testing import CliRunner

import whirld
from whirld.cli import app
from whirld.demo.runner import DemoOptions, DemoReport, DemoRunner, DemoRuntimeIdentity
from whirld.demo.sample import BundledDemoSample
from whirld.errors import InvalidInputError
from whirld.onboarding import doctor
from whirld.onboarding.doctor import DependencyCheck, DiagnosticsService, DoctorReport


@pytest.fixture
def healthy_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real diagnostics and filesystem checks while isolating optional imports."""

    def dependency(distribution: str, module: str) -> DependencyCheck:
        """Report a successful selected import without loading a model framework."""
        return DependencyCheck(
            distribution=distribution,
            module=module,
            version="1.0",
            status="pass",
            message=f"Fixture {module} is available.",
        )

    monkeypatch.setattr(DiagnosticsService, "_dependency", staticmethod(dependency))
    monkeypatch.setattr(
        doctor.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**40)
    )


def test_json_doctor_reports_ready_to_download_without_creating_cache(
    whirld_home: Path, healthy_runtime: None
) -> None:
    """The global CLI callback cannot seed logs/cache before diagnostic execution."""
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = DoctorReport.model_validate_json(result.stdout)
    assert report.ready_for_demo and not report.model_installed
    assert report.model_download_bytes == 408397807
    assert report.cache_path == whirld_home
    assert not whirld_home.exists()


def test_doctor_json_diagnoses_unwritable_cache_before_logging(
    whirld_home: Path, healthy_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable home yields a structured blocker instead of a logger exception."""
    monkeypatch.setattr(doctor.os, "access", lambda path, mode: False)
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 7, result.output
    report = DoctorReport.model_validate_json(result.stdout)
    assert not report.ready_for_demo
    assert (
        next(check for check in report.checks if check.code == "cache").status == "fail"
    )
    assert not whirld_home.exists()


def test_doctor_json_handles_a_regular_file_as_cache_home(
    tmp_path: Path, healthy_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing non-directory state is preserved and remains diagnosable."""
    home = tmp_path / "home-file"
    home.write_bytes(b"existing user file")
    monkeypatch.setenv("WHIRLD_HOME", str(home))
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 7, result.output
    report = DoctorReport.model_validate_json(result.stdout)
    assert not report.ready_for_demo and report.cache_path == home
    assert home.read_bytes() == b"existing user file"
    assert list(tmp_path.iterdir()) == [home]


def test_human_doctor_explains_cpu_and_first_weight_download(
    whirld_home: Path, healthy_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy new user sees the next acquisition and selected execution device."""
    monkeypatch.setenv("COLUMNS", "400")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    text = unstyle(result.output)
    compact = "".join(text.split())
    assert "Ready for CPU inference" in " ".join(text.split())
    assert "408.4 MB" in " ".join(text.split()) and "demo" in text
    assert "".join(sys.executable.split()) in compact
    assert "".join(str(whirld_home).split()) in compact
    assert not whirld_home.exists()


def test_human_doctor_missing_runtime_prints_active_environment_repair(
    whirld_home: Path, healthy_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair command addresses the failing interpreter and is safe to copy."""

    def missing(distribution: str, module: str) -> DependencyCheck:
        """Simulate one absent required runtime with healthy base dependencies."""
        return DependencyCheck(
            distribution=distribution,
            module=module,
            version=None if module == "torch" else "1.0",
            status="fail" if module == "torch" else "pass",
            message="Torch is missing" if module == "torch" else "Available",
        )

    monkeypatch.setattr(DiagnosticsService, "_dependency", staticmethod(missing))
    monkeypatch.setenv("COLUMNS", "400")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 7, result.output
    text = unstyle(result.output)
    compact = "".join(text.split())
    assert "Torch is missing" in " ".join(text.split())
    assert "".join(sys.executable.split()) in compact
    assert "requirements-quickstart.txt" in compact
    assert "Ready for CPU inference" not in " ".join(text.split())
    assert not whirld_home.exists()


def test_unknown_model_doctor_preserves_domain_exit_code(
    whirld_home: Path, healthy_runtime: None
) -> None:
    """Unknown selections fail before cache writes and retain the standard CLI code."""
    result = CliRunner().invoke(app, ["doctor", "--model", "unknown-model"])
    assert result.exit_code == 2 and "No bundled model" in result.output
    assert not whirld_home.exists()


@pytest.mark.parametrize("command", [[], ["doctor"], ["demo"]])
def test_installed_help_in_fresh_process_does_not_import_model_frameworks(
    tmp_path: Path, command: list[str]
) -> None:
    """Every first-run help surface stays usable before optional runtimes exist."""
    code = (
        "import json,sys; from typer.testing import CliRunner; "
        "from whirld.cli import app; "
        "result=CliRunner().invoke(app,json.loads(sys.argv[1])+['--help']); "
        "assert result.exit_code==0,result.output; "
        "assert not {'torch','open_clip','terratorch'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(command)],
        env={**os.environ, "WHIRLD_HOME": str(tmp_path / "child-home")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def fake_demo_report(options: DemoOptions) -> DemoReport:
    """Return typed result metadata at the isolated workflow boundary."""
    output = options.output_dir.resolve()
    return DemoReport(
        model_version="1.0.0",
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        registry_sha256="c" * 64,
        sample=BundledDemoSample.load(),
        output_dir=output,
        scores_path=output / "scores.geojson",
        preview_path=output / "preview.html",
        feature_count=4,
        threads=options.threads,
        elapsed_seconds=1.0,
        runtime=DemoRuntimeIdentity(
            python="3.13.7",
            platform="fixture",
            whirld="0.3.0",
            torch="2.12.1",
            open_clip="3.3.0",
            numpy="2.5.0",
            rasterio="1.5.2",
        ),
    )


def test_demo_command_forwards_options_and_prints_result_paths_without_markup(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed command output tells users where to inspect a real-run result."""
    observed: list[DemoOptions] = []

    def run(instance: DemoRunner) -> DemoReport:
        """Observe typed options; the library's own tests cover actual execution."""
        observed.append(instance.options)
        return fake_demo_report(instance.options)

    monkeypatch.setattr(DemoRunner, "run", run)
    monkeypatch.setenv("COLUMNS", "400")
    output = tmp_path / "folder [red]literal"
    result = CliRunner().invoke(
        app, ["demo", "--output", str(output), "--threads", "3"]
    )
    assert result.exit_code == 0, result.output
    assert observed == [DemoOptions(output_dir=output, threads=3)]
    text = unstyle(result.output)
    compact = "".join(text.split())
    assert "Scored 4 image chips" in " ".join(text.split())
    assert "".join(str(output / "preview.html").split()) in compact
    assert "".join(str(output / "scores.geojson").split()) in compact
    assert "not land-cover confidence" in " ".join(text.split())


@pytest.mark.parametrize("error", [InvalidInputError, OSError])
def test_demo_setup_or_storage_failure_is_an_actionable_domain_error(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
) -> None:
    """Expected setup failures map to a documented status without a traceback."""

    def run(instance: DemoRunner) -> DemoReport:
        """Raise at the library boundary after its resource owner handles cleanup."""
        raise error("fixture setup/storage failure")

    monkeypatch.setattr(DemoRunner, "run", run)
    result = CliRunner().invoke(app, ["demo", "--output", str(tmp_path / "result")])
    assert result.exit_code == 7 and "fixture setup/storage failure" in result.output
    assert not (tmp_path / "result").exists()
    assert "Traceback" not in result.output


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_demo_rejects_invalid_threads_before_invoking_runner(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch, threads: str
) -> None:
    """Bad CLI settings cannot initiate optional imports, downloads or inference."""

    def unexpected(instance: DemoRunner) -> DemoReport:
        """Fail if invalid argument handling reaches the expensive workflow."""
        raise AssertionError("runner must not execute")

    monkeypatch.setattr(DemoRunner, "run", unexpected)
    result = CliRunner().invoke(app, ["demo", "--threads", threads], color=True)
    assert result.exit_code == 2, result.output
    assert "--threads" in unstyle(result.output)


def test_direct_demo_diagnoses_non_directory_cache_before_file_logging(
    tmp_path: Path, healthy_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct first-run command reaches useful preflight before logger writes."""
    home = tmp_path / "cache-file"
    home.write_bytes(b"existing user file")
    output = tmp_path / "new-result"
    monkeypatch.setenv("WHIRLD_HOME", str(home))

    def unexpected(*args: object, **kwargs: object) -> None:
        """Fail if a storage blocker reaches model acquisition."""
        raise AssertionError("must not download")

    monkeypatch.setattr(whirld, "pull", unexpected)
    result = CliRunner().invoke(app, ["demo", "--output", str(output)])
    assert result.exit_code == 7, result.output
    assert "cannot be written" in " ".join(unstyle(result.output).split())
    assert home.read_bytes() == b"existing user file" and not output.exists()
    assert list(tmp_path.iterdir()) == [home]


@pytest.mark.parametrize(
    ("flags", "verbose", "quiet"),
    [
        (["--verbose"], True, False),
        (["--quiet"], False, True),
        (["--verbose", "--quiet"], True, True),
    ],
)
def test_demo_retains_root_logging_flags_after_deferred_file_setup(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    verbose: bool,
    quiet: bool,
) -> None:
    """Delaying filesystem logging does not discard explicitly selected verbosity."""
    observed: list[DemoOptions] = []

    def run(instance: DemoRunner) -> DemoReport:
        """Capture typed execution flags before the library applies its logger setup."""
        observed.append(instance.options)
        return fake_demo_report(instance.options)

    monkeypatch.setattr(DemoRunner, "run", run)
    output = tmp_path / "result"
    result = CliRunner().invoke(app, [*flags, "demo", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert len(observed) == 1
    assert observed[0].verbose is verbose and observed[0].quiet is quiet
    assert not whirld_home.exists()


@pytest.mark.parametrize(
    "arguments",
    [["--help"], ["doctor", "--help"], ["demo", "--help"], ["doctor", "--json"]],
)
def test_help_and_doctor_work_when_rasterio_native_import_is_broken(
    tmp_path: Path, arguments: list[str]
) -> None:
    """Diagnose a native dependency failure before importing the inference API."""
    code = """
import json, sys
from types import SimpleNamespace
class BlockRasterio:
    def find_spec(self, name, path=None, target=None):
        if name == 'rasterio' or name.startswith('rasterio.'):
            raise ImportError('libexpat.so.1: cannot open shared object file')
sys.meta_path.insert(0, BlockRasterio())
from typer.testing import CliRunner
from whirld.cli import app
arguments = json.loads(sys.argv[1])
if '--json' in arguments:
    from whirld.onboarding import doctor
    original = doctor.DiagnosticsService._dependency
    def dependency(package, module):
        if module == 'rasterio':
            return original(package, module)
        return doctor.DependencyCheck(distribution=package,module=module,
            version='1.0',status='pass',message='Fixture runtime ready')
    doctor.DiagnosticsService._dependency = staticmethod(dependency)
    doctor.platform.system = lambda: 'Linux'
    doctor.shutil.disk_usage = lambda path: SimpleNamespace(free=2**40)
result = CliRunner().invoke(app, arguments)
assert result.exit_code == (7 if '--json' in arguments else 0), result.output
assert 'whirld.api' not in sys.modules
if '--json' in arguments:
    report = json.loads(result.stdout)
    check = next(row for row in report['checks'] if row['code']=='dependency:rasterio')
    assert check['status'] == 'fail'
    assert 'libexpat1' in check['remedy']
    assert 'pip install' not in check['remedy']
    assert not report['ready_for_demo']
assert 'Traceback' not in result.output
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(arguments)],
        env={**os.environ, "WHIRLD_HOME": str(tmp_path / "fresh-home")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "fresh-home").exists()
