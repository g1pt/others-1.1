from src.analysis import SummaryRow, rank_summaries, select_candidates


def test_select_candidates_and_ranking() -> None:
    rows = [
        SummaryRow(
            key="A",
            trades=30,
            winrate=0.5,
            expectancy=0.2,
            max_drawdown=8,
            stability=0.3,
        ),
        SummaryRow(
            key="B",
            trades=40,
            winrate=0.6,
            expectancy=0.2,
            max_drawdown=6,
            stability=0.2,
        ),
        SummaryRow(
            key="C",
            trades=28,
            winrate=0.4,
            expectancy=0.2,
            max_drawdown=6,
            stability=0.9,
        ),
        SummaryRow(
            key="D",
            trades=50,
            winrate=0.7,
            expectancy=0.4,
            max_drawdown=10,
            stability=0.5,
        ),
        SummaryRow(
            key="E",
            trades=24,
            winrate=0.7,
            expectancy=0.3,
            max_drawdown=5,
            stability=0.6,
        ),
        SummaryRow(
            key="F",
            trades=60,
            winrate=0.2,
            expectancy=-0.1,
            max_drawdown=4,
            stability=0.8,
        ),
    ]

    filtered = select_candidates(rows, min_trades=25, min_expectancy=0.0, max_dd=12)
    assert [row.key for row in filtered] == ["A", "B", "C", "D"]

    ranked = rank_summaries(filtered)
    assert [row.key for row in ranked] == ["D", "C", "B", "A"]
