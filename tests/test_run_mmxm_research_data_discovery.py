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
