import pandas as pd

from ledger_htr.predict_vlm import load_partial_predictions


def test_load_partial_predictions_missing_file_returns_empty(tmp_path):
    assert load_partial_predictions(str(tmp_path / "nonexistent.csv")) == {}


def test_load_partial_predictions_reads_existing_csv(tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame({"ID": ["a1", "a2"], "Target": ["hello", "world"]}).to_csv(path, index=False)
    result = load_partial_predictions(str(path))
    assert result == {"a1": "hello", "a2": "world"}
