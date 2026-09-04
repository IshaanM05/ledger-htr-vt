import csv
import dataclasses
import datetime
import json
import os


class RunLogger:
    """Lightweight local experiment logger: one dir per run under `runs/`,
    with the resolved config and a metrics.csv appended to during training.
    No external dependencies (tensorboard/wandb are optional add-ons, not
    required to get basic reproducibility records).
    """

    def __init__(self, runs_dir: str, run_name: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(runs_dir, f"{run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self._metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self._metrics_fields: list[str] | None = None

    def log_config(self, cfg) -> None:
        cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)

    def log_metrics(self, step: int, **metrics) -> None:
        row = {"step": step, **metrics}
        is_new_file = not os.path.exists(self._metrics_path)
        if self._metrics_fields is None:
            self._metrics_fields = list(row.keys())
        with open(self._metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metrics_fields)
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
