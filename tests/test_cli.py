import sys

from adaptive_hitl_agent.cli import build_parser, main


def test_parser_exposes_experiment_defaults() -> None:
    args = build_parser().parse_args(["experiment"])
    assert args.episodes == 600
    assert args.seed == 7
    assert args.output_dir == "artifacts/latest"


def test_cli_runs_experiment_and_writes_artifacts(tmp_path, monkeypatch, capsys) -> None:
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaptive-hitl",
            "experiment",
            "--episodes",
            "10",
            "--seed",
            "3",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    captured = capsys.readouterr().out
    assert "adaptive_rl" in captured
    assert "Artifacts written to" in captured
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "traces.jsonl").exists()
    assert (output_dir / "dqn_policy.pt").exists()


def test_cli_rejects_zero_episodes(monkeypatch, capsys) -> None:
    import pytest

    monkeypatch.setattr(sys, "argv", ["adaptive-hitl", "experiment", "--episodes", "0"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "must be positive" in capsys.readouterr().err
