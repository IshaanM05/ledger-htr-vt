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

    data_dir: str = os.path.join(REPO_ROOT, "data", "raw")
    image_dir: str = os.path.join(REPO_ROOT, "data", "raw", "images")
    folds_csv: str = os.path.join(REPO_ROOT, "data", "processed", "folds.csv")
    checkpoint_dir: str = os.path.join(REPO_ROOT, "checkpoints")
    runs_dir: str = os.path.join(REPO_ROOT, "runs")

    extra: dict = dataclasses.field(default_factory=dict)


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    known, extra = {}, {}
    for k, v in raw.items():
        (known if k in known_fields else extra)[k] = v
    if extra:
        known["extra"] = {**extra, **known.get("extra", {})}
    return ExperimentConfig(**known)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
