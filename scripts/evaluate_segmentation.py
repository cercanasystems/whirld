"""Run the fixed labeled burn-scar baseline without training or threshold tuning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from whirld.errors import WhirldError
from whirld.evaluation.runner import EvaluationRunner, SuiteLoader
from whirld.evaluation.schemas import EvaluationOptions
from whirld.logging_setup import configure_logging


class EvaluationCommand:
    """Adapt CLI arguments into the typed fixed-protocol evaluation runner."""

    def run(self, argv: list[str] | None = None) -> int:
        """Evaluate selected split(s), logging the final report location or failure.

        Args:
            argv: Optional explicit argument list; None uses process arguments.

        Returns:
            Zero on success, a domain exit code or one for invalid local settings.

        Raises:
            SystemExit: Argparse help or malformed CLI arguments.
            KeyboardInterrupt: Interruption after the runner cleans owned outputs.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument(
            "--cache-dir", type=Path, default=Path("evaluation-runs/cache")
        )
        parser.add_argument("--manifest", type=Path)
        parser.add_argument(
            "--split", choices=("validation", "test", "all"), default="validation"
        )
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--cache-mb", type=int, default=8)
        parser.add_argument("--threads", type=int, default=2)
        parser.add_argument("--offline", action="store_true")
        args = parser.parse_args(argv)
        configure_logging()
        logger = logging.getLogger(__name__)
        try:
            options = EvaluationOptions(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                split=args.split,
                batch_size=args.batch_size,
                cache_mb=args.cache_mb,
                threads=args.threads,
                offline=args.offline,
            )
            report = EvaluationRunner(SuiteLoader.load(args.manifest), options).run()
            for split in report.splits:
                logger.info("%s: %s", split.split, split.metrics.model_dump_json())
            logger.info("Report: %s", options.output_dir.resolve() / "report.json")
            return 0
        except WhirldError as exc:
            logger.error("%s", exc)
            return exc.exit_code
        except (OSError, ValidationError) as exc:
            logger.error("Evaluation failed: %s", exc)
            return 1


if __name__ == "__main__":
    raise SystemExit(EvaluationCommand().run())
