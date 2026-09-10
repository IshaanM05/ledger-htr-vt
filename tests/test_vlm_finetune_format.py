import json

import pandas as pd

from ledger_htr.data.vlm_finetune_format import build_conversations, write_internvl_jsonl, write_meta_json


def test_build_conversations_shape():
    conv = build_conversations("Hello world")
    assert conv == [
        {"from": "human", "value": conv[0]["value"]},
        {"from": "gpt", "value": "Hello world"},
    ]
    assert conv[0]["value"].startswith("<image>\n")


def test_write_internvl_jsonl_excludes_corrupted(tmp_path):
    from ledger_htr.data.known_issues import CORRUPTED_TRAIN_IDS

    corrupted_id = next(iter(CORRUPTED_TRAIN_IDS))
    df = pd.DataFrame({"ID": [corrupted_id, "clean1", "clean2"], "Target": ["bad", "good one", "good two"]})
    out_path = str(tmp_path / "train.jsonl")

    n = write_internvl_jsonl(df, ".", out_path, exclude_corrupted=True)
    assert n == 2

    with open(out_path) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert all(entry["conversations"][1]["value"] != "bad" for entry in lines)
    assert lines[0]["image"] == "./clean1.jpg"


def test_write_internvl_jsonl_keeps_corrupted_when_disabled(tmp_path):
    from ledger_htr.data.known_issues import CORRUPTED_TRAIN_IDS

    corrupted_id = next(iter(CORRUPTED_TRAIN_IDS))
    df = pd.DataFrame({"ID": [corrupted_id], "Target": ["bad"]})
    out_path = str(tmp_path / "val.jsonl")

    n = write_internvl_jsonl(df, ".", out_path, exclude_corrupted=False)
    assert n == 1


def test_write_meta_json(tmp_path):
    out_path = str(tmp_path / "meta.json")
    write_meta_json("my_ds", "/abs/train.jsonl", "/abs/images", 42, out_path)
    with open(out_path) as f:
        meta = json.load(f)
    assert meta["my_ds"]["length"] == 42
    assert meta["my_ds"]["annotation"] == "/abs/train.jsonl"
