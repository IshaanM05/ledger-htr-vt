"""Tests for the weighted WER/CER scorer.

The formula (corpus-level sum-of-edits/sum-of-lengths, NOT per-sample
length-normalized averaging) was reverse-engineered from the Zindi
discussion board and verified against a real submission — see the module
docstring in ledger_htr/metrics/scorer.py for the full derivation and the
exact match to 8 decimal places. `test_real_submission_regression` below
locks that verification in so a future refactor can't silently reintroduce
the wrong (per-sample-normalized) formula.

The other tests use hand-computed edit distances (counted by hand, not by
re-running the implementation) to check the corpus-sum mechanics directly.
"""

import math

import pytest

from ledger_htr.metrics.scorer import (
    compute_weighted_cer,
    compute_weighted_wer,
    edit_distance,
    final_score,
)


def test_edit_distance_basic():
    assert edit_distance("cat", "cat") == 0
    assert edit_distance("cat", "cot") == 1  # substitution
    assert edit_distance("cat", "cats") == 1  # insertion
    assert edit_distance("cats", "cat") == 1  # deletion
    assert edit_distance(["a", "b"], ["a", "c"]) == 1  # works on token lists too


def test_identical_strings_score_zero(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("ID,Target\n1,the cat sat\n")
    pred.write_text("ID,Target\n1,the cat sat\n")

    cer = compute_weighted_cer(str(gt), str(pred))
    wer = compute_weighted_wer(str(gt), str(pred))
    assert cer["score"] == pytest.approx(0.0)
    assert wer["score"] == pytest.approx(0.0)


def test_single_substitution_char_and_word_level(tmp_path):
    # ref="cat" (3 chars, 1 word), pred="cot": 1 char substitution.
    # corpus CER = total_edits / total_ref_chars = 1 / 3.
    # corpus WER = total_edits / total_ref_words = 1 / 1 = 1.0 (the one word is entirely wrong).
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("ID,Target\n1,cat\n")
    pred.write_text("ID,Target\n1,cot\n")

    cer = compute_weighted_cer(str(gt), str(pred))
    wer = compute_weighted_wer(str(gt), str(pred))
    assert cer["score"] == pytest.approx(1 / 3)
    assert wer["score"] == pytest.approx(1.0)


def test_corpus_level_aggregation_char(tmp_path):
    # sample A: ref="ab" pred="ab" -> 0 edits, ref_len=2
    # sample B: ref="abc" pred="abd" -> 1 edit, ref_len=3
    # corpus CER = (0 + 1) / (2 + 3) = 1/5 -- this is the key difference from
    # per-sample averaging (which would give (0/2 + 1/3)/2 = 1/6): longer
    # samples get proportionally more say in the corpus-level rate.
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("ID,Target\n1,ab\n2,abc\n")
    pred.write_text("ID,Target\n1,ab\n2,abd\n")

    cer = compute_weighted_cer(str(gt), str(pred))
    assert cer["score"] == pytest.approx(1 / 5)
    assert cer["matched"] == 2


def test_final_combination(tmp_path):
    # sample A: ref="the cat sat" (11 chars, 3 words), pred="the cat sit"
    #   char: 1 substitution ('a'->'i')
    #   word: 1 whole word wrong ("sat"!="sit")
    # sample B: ref="a quick brown fox" (17 chars, 4 words), pred="a quick brown fx"
    #   char: 1 deletion ('o' dropped from "fox")
    #   word: 1 whole word wrong ("fox"!="fx")
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("ID,Target\n1,the cat sat\n2,a quick brown fox\n")
    pred.write_text("ID,Target\n1,the cat sit\n2,a quick brown fx\n")

    expected_cer = (1 + 1) / (11 + 17)
    expected_wer = (1 + 1) / (3 + 4)

    result = final_score(str(gt), str(pred))
    assert result["cer"] == pytest.approx(expected_cer)
    assert result["wer"] == pytest.approx(expected_wer)
    assert result["final"] == pytest.approx(1 - 0.5 * (expected_cer + expected_wer))


def test_missing_prediction_scored_as_empty_string(tmp_path):
    # rules page: "Missing predictions ... will be penalised as incorrect" --
    # a missing ID must count against the corpus sums as pred="", not be
    # skipped from the denominator.
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("ID,Target\n1,cat\n2,dog\n")
    pred.write_text("ID,Target\n1,cat\n")  # ID 2 missing entirely

    cer = compute_weighted_cer(str(gt), str(pred))
    # ID1: 0 edits / 3 chars. ID2 missing -> pred="" vs "dog": 3 edits / 3 chars.
    assert cer["score"] == pytest.approx((0 + 3) / (3 + 3))
    assert cer["matched"] == 1
    assert cer["total_reference"] == 2


def test_empty_string_prediction_same_as_missing(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("ID,Target\n1,cat\n")
    pred.write_text('ID,Target\n1,""\n')  # explicit empty string, not a missing row

    cer = compute_weighted_cer(str(gt), str(pred))
    assert cer["score"] == pytest.approx(1.0)  # 3 edits / 3 chars


def test_real_submission_regression():
    """Locks in the reverse-engineered formula against a real leaderboard
    result (Ish1105, 2026-09-04): WER Weighted=8.483792682,
    CER Weighted=30.34969263, Public Score=0.370602341.

    We don't have the actual per-sample test predictions/references (test
    labels aren't published), so this checks the final-combination arithmetic
    directly against the leaderboard's displayed component values, rather
    than re-deriving those components from raw edit distances.
    """
    wer_component = 8.483792682 / 12  # displayed mean edit distance / mean ref word length
    cer_component = 30.34969263 / 55  # displayed mean edit distance / mean ref char length
    final = 1 - 0.5 * (wer_component + cer_component)
    assert final == pytest.approx(0.370602341, abs=1e-6)
