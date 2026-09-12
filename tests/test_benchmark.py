import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"


def test_benchmark_writes_runs_and_summary(tmp_path) -> None:
    output_dir = tmp_path / "results"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--episodes", "5", "--seeds", "1", "7",
         "--output-dir", str(output_dir), "--artifacts-dir", str(tmp_path / "artifacts")],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output_dir / "benchmark_multiseed.json").read_text())
    assert set(summary["runs"]) == {"1", "7"}
    assert summary["seeds"] == [1, 7]
    assert "sample_std" in summary["summary"]["adaptive_rl"]["average_reward"]
    assert (output_dir / "benchmark_seed7.json").exists()


def test_benchmark_rejects_repeated_seeds(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "7", "7", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "seeds must be distinct" in result.stderr


def test_reference_seed_is_configurable(tmp_path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--episodes", "1", "--seeds", "3",
         "--reference-seed", "3", "--output-dir", str(tmp_path / "results"),
         "--artifacts-dir", str(tmp_path / "artifacts")],
        check=True, capture_output=True, text=True,
    )
    output = tmp_path / "results"
    assert (output / "benchmark_seed3.json").exists()
    assert not (output / "benchmark_seed7.json").exists()
    payload = json.loads((output / "benchmark_multiseed.json").read_text())
    assert payload["reference_seed"] == 3
    assert payload["summary"]["adaptive_rl"]["average_reward"]["sample_std"] == 0
