"""Local implementation of the competition's scoring metric.

The formula below was reverse-engineered from the Zindi discussion board
("Help Please", thread 33847 — user J0NNY, 4 upvotes, no organizer
correction) and independently verified against our own real submission
(Ish1105, 2026-09-04): the leaderboard displayed WER Weighted=8.483792682,
CER Weighted=30.34969263, Public Score=0.370602341, and
1 - 0.5*(8.483792682/12 + 30.34969263/55) = 0.370602342 — matches to 8
decimal places.

The key realization: "WER Weighted"/"CER Weighted" are NOT per-sample
length-normalized rates averaged across samples (which is what the original
version of this scorer assumed, following the generic Zindi weighted-metric
guide and Starters.zip's eval_metrics.py defaults — both wrong). They're
corpus-level (micro-averaged) error rates: sum of all edit distances across
every sample divided by the sum of all reference lengths across every
sample. This is algebraically identical to mean(edit_distance) /
mean(reference_length) over the same sample set, which is why J0NNY's
snippet divided the leaderboard's displayed mean-edit-distance figures by
flat constants (12, 55) — those constants are just the mean reference
word/char length of whatever split was being scored, not universal magic
numbers. Computing sum(edits)/sum(lengths) directly (as this module does)
is the general, dataset-agnostic form of the same formula and needs no
hardcoded constants: it self-normalizes to whatever set it's scoring
(a local val fold, the full train set, etc.), which is exactly what we want.

This also explains the rules page's plain-English description ("Longer
reference transcriptions are weighted more heavily") precisely: a
corpus-level sum-of-edits/sum-of-lengths average gives longer samples more
influence over the total than a macro-average of per-sample rates would.

Per the official rules page (Evaluation section): final score combines CER
and WER with equal 0.5/0.5 weight. Per the rules page: "Missing predictions,
empty predictions, or invalid text values will be penalised as incorrect" —
implemented here as scoring a missing/empty prediction against its reference
as if the prediction were an empty string (maximal edit distance), rather
than excluding it from the corpus sums.
"""

import pandas as pd


def edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein distance over any sequence (chars or word tokens)."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def _corpus_error_rate(preds: dict[str, str], refs: dict[str, str], level: str) -> dict:
    """level: 'char' or 'word'. Missing/empty predictions score against the
    reference as an empty string (rules page: penalised as incorrect, not excluded)."""
    total_edits = 0
    total_ref_len = 0
    n_missing = 0

    for id_, ref in refs.items():
        pred = preds.get(id_, "")
        if id_ not in preds or not isinstance(pred, str) or pred.strip() == "":
            n_missing += 1
            pred = ""

        ref_seq = ref if level == "char" else ref.split()
        pred_seq = pred if level == "char" else pred.split()
        total_edits += edit_distance(pred_seq, ref_seq)
        total_ref_len += len(ref_seq)

    return {
        "n": len(refs),
        "n_missing": n_missing,
        "rate": (total_edits / total_ref_len) if total_ref_len > 0 else 0.0,
    }


def _load_id_target(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    ids = df["ID"].astype(str)
    # NaN Target (blank CSV cell) must become "" here, not the literal string
    # "nan" that .astype(str) would otherwise produce on a float NaN.
    targets = df["Target"].where(df["Target"].notna(), "").astype(str)
    return dict(zip(ids, targets))


def compute_weighted_cer(gt_csv_path: str, pred_csv_path: str) -> dict:
    refs = _load_id_target(gt_csv_path)
    preds = _load_id_target(pred_csv_path)
    result = _corpus_error_rate(preds, refs, level="char")
    return {"matched": result["n"] - result["n_missing"], "total_reference": result["n"], "score": result["rate"]}


def compute_weighted_wer(gt_csv_path: str, pred_csv_path: str) -> dict:
    refs = _load_id_target(gt_csv_path)
    preds = _load_id_target(pred_csv_path)
    result = _corpus_error_rate(preds, refs, level="word")
    return {"matched": result["n"] - result["n_missing"], "total_reference": result["n"], "score": result["rate"]}


def final_score_from_dicts(preds: dict[str, str], refs: dict[str, str], cer_weight: float = 0.5, wer_weight: float = 0.5) -> dict:
    """In-memory version of final_score, for scoring model predictions directly
    (e.g. once per validation epoch) without round-tripping through CSV files."""
    cer_result = _corpus_error_rate(preds, refs, level="char")
    wer_result = _corpus_error_rate(preds, refs, level="word")

    final = 1 - (cer_weight * cer_result["rate"] + wer_weight * wer_result["rate"])
    return {
        "cer": cer_result["rate"],
        "wer": wer_result["rate"],
        "final": final,
        "matched": cer_result["n"] - cer_result["n_missing"],
        "total_reference": cer_result["n"],
    }


def final_score(gt_csv_path: str, pred_csv_path: str, cer_weight: float = 0.5, wer_weight: float = 0.5) -> dict:
    refs = _load_id_target(gt_csv_path)
    preds = _load_id_target(pred_csv_path)
    return final_score_from_dicts(preds, refs, cer_weight=cer_weight, wer_weight=wer_weight)
