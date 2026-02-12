from pathlib import Path

from scripts import run_mmxm_research as mod


def test_canonical_dataset_key_strips_windows_copy_suffix() -> None:
    key = mod._canonical_dataset_key(Path("FX_SPX500, 30 (2).csv"))
    assert key == "fx_spx500, 30"


def test_find_data_files_matches_numbered_copy_names(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    wanted = data_dir / "FX_SPX500, 30 (2).csv"
    wanted.write_text("timestamp,open,high,low,close\n", encoding="utf-8")
    other = data_dir / "FX_SPX500, 1.csv"
    other.write_text("timestamp,open,high,low,close\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_data_roots", lambda: [data_dir])

    files = mod._find_data_files(False, "FX_SPX500", [30])

    assert files == [wanted]


def test_resolve_data_files_falls_back_to_all_datasets(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fallback = data_dir / "FX_SPX500, 1.csv"
    fallback.write_text("timestamp,open,high,low,close\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_data_roots", lambda: [data_dir])

    files, used_fallback = mod._resolve_data_files(False, "FX_SPX500", [30])

    assert files == [fallback]
    assert used_fallback is True


def test_resolve_data_files_without_fallback_when_default_match_exists(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    default = data_dir / "FX_SPX500, 30.csv"
    default.write_text("timestamp,open,high,low,close\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_data_roots", lambda: [data_dir])

    files, used_fallback = mod._resolve_data_files(False, "FX_SPX500", [30])

    assert files == [default]
    assert used_fallback is False


def test_step2_overrides_from_cli_are_threaded_into_run(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    default = data_dir / "FX_SPX500, 30.csv"
    default.write_text("timestamp,open,high,low,close\n", encoding="utf-8")

    captured: dict = {}

    monkeypatch.setattr(mod, "_data_roots", lambda: [data_dir])

    def _fake_run(path, combo_filter, initial_equity, risk_pct, max_dd_pct, max_trades_per_day, paper_execute, step2_overrides):
        captured["path"] = path
        captured["paper_execute"] = paper_execute
        captured["step2_overrides"] = step2_overrides

    monkeypatch.setattr(mod, "_run_instrument_step4c", _fake_run)

    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: mod.argparse.Namespace(
            combo_filter=None,
            all_datasets=False,
            symbol="FX_SPX500",
            tfs=[30],
            initial_equity=10000.0,
            risk=0.02,
            max_dd=0.03,
            daily_loss=0.02,
            max_trades_day=3,
            baseline=False,
            self_test=False,
            paper_execute=True,
            paper_risk_pct=0.03,
            paper_max_trades_day=2,
            paper_stop_after_losses=1,
            paper_daily_dd=0.03,
            paper_hard_dd=0.05,
        ),
    )

    mod.main()

    assert captured["path"] == default
    assert captured["paper_execute"] is True
    assert captured["step2_overrides"] == {
        "risk_per_trade_pct": 0.03,
        "max_trades_per_day": 2,
        "stop_after_consecutive_losses": 1,
        "daily_drawdown_stop_pct": 0.03,
        "hard_max_drawdown_pct": 0.05,
    }
