import json

import app.main as main


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, dict):
                handle.write(json.dumps(row) + "\n")
            else:
                handle.write(str(row) + "\n")


def test_status_uses_custom_limit_and_includes_rejections(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)

    _write_jsonl(
        tmp_path / "paper_trades.log",
        [{"id": f"t{i}"} for i in range(1, 6)],
    )
    _write_jsonl(
        tmp_path / "paper_orders.log",
        [{"id": f"o{i}"} for i in range(1, 4)],
    )
    _write_jsonl(
        tmp_path / "rejections.log",
        [{"reason": "risk_limit"}, {"reason": "bad_secret"}],
    )

    body = main.status(limit=2)

    assert body["limit"] == 2
    assert body["trades_logged"] == 2
    assert body["orders_logged"] == 2
    assert body["rejections_logged"] == 2
    assert body["last_trades"] == [{"id": "t4"}, {"id": "t5"}]
    assert body["last_orders"] == [{"id": "o2"}, {"id": "o3"}]
    assert body["last_rejections"] == [{"reason": "risk_limit"}, {"reason": "bad_secret"}]


def test_tail_jsonl_ignores_invalid_lines(tmp_path):
    _write_jsonl(
        tmp_path / "paper_trades.log",
        ["not-json", {"id": "t1"}],
    )

    trades = main._tail_jsonl(tmp_path / "paper_trades.log")
    assert trades == [{"id": "t1"}]
