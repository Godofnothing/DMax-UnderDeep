#!/usr/bin/env python3
"""Publish the final postprocessed accuracy and dInfer speed results to Underdeep.

The evaluator writes the speed summary to stdout and the task-specific ``val_*.py``
script writes the authoritative accuracy to a second log. Keeping this outside
the model worker also ensures only one process creates an Underdeep run for
tensor-parallel evaluations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


SPEED_RE = re.compile(
    r"Forward:\s*(?P<Forward>\d+)\s*,\s*"
    r"Time:\s*(?P<Time>[-+0-9.eE]+)\s*,\s*"
    r"FPS:\s*(?P<FPS>[-+0-9.eE]+)\s*,\s*"
    r"TPS:\s*(?P<TPS>[-+0-9.eE]+)\s*,\s*"
    r"TPF:\s*(?P<TPF>[-+0-9.eE]+)"
)
ACCURACY_RE = re.compile(r"^Accuracy:\s*(?P<Accuracy>[-+0-9.eE]+)%\s*$", re.MULTILINE)


def _metric_key(value: str) -> str:
    """Make a stable Underdeep path component from an lm-eval table cell."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")


def _read_metrics(log_path: Path, score_log_path: Path | None = None) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(SPEED_RE.finditer(text))
    if not matches:
        raise ValueError(f"No dInfer speed summary found in {log_path}")

    values: dict[str, float | int] = {}
    for name, value in matches[-1].groupdict().items():
        parsed: float | int = int(value) if name == "Forward" else float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Invalid {name} value in {log_path}: {value}")
        values[name] = parsed

    if score_log_path is not None:
        score_text = score_log_path.read_text(encoding="utf-8", errors="replace")
        score_matches = list(ACCURACY_RE.finditer(score_text))
        if not score_matches:
            raise ValueError(f"No postprocessed accuracy found in {score_log_path}")
        accuracy = float(score_matches[-1].group("Accuracy")) / 100.0
        if not math.isfinite(accuracy):
            raise ValueError(f"Invalid Accuracy value in {score_log_path}")
        values["Accuracy"] = accuracy
        return values

    # lm-eval prints rows in a Markdown table.  Column 0 can be empty when it
    # repeats the previous task, hence current_task is retained across rows.
    current_task = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        columns = [cell.strip() for cell in line[1:-1].split("|")]
        if len(columns) != 9 or columns[0] == "Tasks" or set("".join(columns)) <= {"-", ":"}:
            continue
        if columns[0]:
            current_task = columns[0]
        task, filter_name, metric = current_task, columns[2], columns[4]
        if not task or not filter_name or not metric:
            continue
        try:
            value, stderr = float(columns[6]), float(columns[8])
        except ValueError:
            continue
        prefix = "/".join((_metric_key(task), _metric_key(filter_name), _metric_key(metric)))
        values[prefix] = value
        values[f"{prefix}_stderr"] = stderr

    if not any("/" in name for name in values):
        raise ValueError(f"No lm-eval metric rows found in {log_path}")
    return values


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="dllm")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument(
        "--score-log",
        type=Path,
        default=None,
        help="Log produced by the task-specific val_*.py postprocessor.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--parameters-json", default="{}")
    args = parser.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"Evaluation log does not exist: {args.log}")
    if args.score_log is not None and not args.score_log.is_file():
        raise SystemExit(f"Postprocessing log does not exist: {args.score_log}")
    try:
        parameters: dict[str, Any] = json.loads(args.parameters_json)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--parameters-json must be a JSON object: {error}") from error
    if not isinstance(parameters, dict):
        raise SystemExit("--parameters-json must be a JSON object")

    metrics = _read_metrics(args.log, args.score_log)
    revision = _git_revision()
    parameters.update({"task": args.task, "evaluation_log": str(args.log)})
    if args.score_log is not None:
        parameters["postprocessing_log"] = str(args.score_log)
    if revision:
        parameters["git_revision"] = revision
    run_name = args.run_name or f"dmax-{args.task}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"

    try:
        import underdeep
    except ImportError as error:
        raise SystemExit(
            "Underdeep logging requires yandex-underdeep; install "
            "dInfer/evaluations/requirements-underdeep.txt"
        ) from error

    run = underdeep.init_run(
        experiment=args.experiment,
        project=args.project,
        name=run_name,
        parameters=parameters,
        # The evaluation has already completed, so upload its single result
        # point synchronously before marking the Underdeep run as finished.
        log_interval=1,
    )
    try:
        run.log(metrics, step=0, flush=True)
    except BaseException as error:
        run.finish(error=str(error))
        raise
    run.finish()
    print(f"Underdeep run: {run.experiment_link}")


if __name__ == "__main__":
    main()
