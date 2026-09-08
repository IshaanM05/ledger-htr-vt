import dataclasses
import os

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclasses.dataclass
class ExperimentConfig:
    run_name: str
    seed: int = 42
    fold: int = 0
    n_folds: int = 5

    model_name: str = "microsoft/trocr-base-handwritten"
    target_height: int = 64
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 5e-5
    num_epochs: int = 10
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    use_elastic_augment: bool = False

    data_dir: str = os.path.join(REPO_ROOT, "data", "raw")
    image_dir: str = os.path.join(REPO_ROOT, "data", "raw", "images")
    folds_csv: str = os.path.join(REPO_ROOT, "data", "processed", "folds.csv")
    checkpoint_dir: str = os.path.join(REPO_ROOT, "checkpoints")
    runs_dir: str = os.path.join(REPO_ROOT, "runs")

    extra: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class HTRVTConfig:
    """Config for the CTC-based HTR-VT model (src/ledger_htr/train_htrvt.py) --
    architecturally distinct from TrOCR's seq2seq ExperimentConfig (iteration-
    based schedule, SAM optimizer, no tokenizer/processor), kept as its own
    dataclass rather than overloading ExperimentConfig with unrelated fields."""

    run_name: str
    seed: int = 42
    fold: int = 0
    n_folds: int = 5

    target_height: int = 64
    max_width: int = 1024
    train_batch_size: int = 32
    val_batch_size: int = 8
    max_lr: float = 1e-3
    weight_decay: float = 0.5
    warmup_iters: int = 400
    total_iters: int = 8000
    eval_every_iters: int = 200
    print_every_iters: int = 50
    mask_ratio: float = 0.4
    max_span_length: int = 8
    ema_decay: float = 0.9999
    use_elastic_augment: bool = False
    use_bf16: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    data_dir: str = os.path.join(REPO_ROOT, "data", "raw")
    image_dir: str = os.path.join(REPO_ROOT, "data", "raw", "images")
    folds_csv: str = os.path.join(REPO_ROOT, "data", "processed", "folds.csv")
    checkpoint_dir: str = os.path.join(REPO_ROOT, "checkpoints")
    runs_dir: str = os.path.join(REPO_ROOT, "runs")

    extra: dict = dataclasses.field(default_factory=dict)


def load_config(path: str, config_cls=ExperimentConfig):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known_fields = {f.name for f in dataclasses.fields(config_cls)}
    known, extra = {}, {}
    for k, v in raw.items():
        (known if k in known_fields else extra)[k] = v
    if extra:
        known["extra"] = {**extra, **known.get("extra", {})}
    return config_cls(**known)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
