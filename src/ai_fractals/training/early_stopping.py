"""early_stopping.py"""


class EarlyStopping:
    def __init__(self, monitor: str, patience: int, mode: str = "auto"):
        self.monitor = monitor
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.epoch = 0
        self.stopped_epoch = None

        if mode is None or mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

    def check(self, score: float) -> bool:
        self.epoch += 1
        if (
            self.best_score is None
            or (self.mode == "min" and score < self.best_score)
            or (self.mode == "max" and score > self.best_score)
        ):
            self.best_score = score
            self.counter = 0
            return False  # continue training

        self.counter += 1
        if self.counter >= self.patience:
            self.stopped_epoch = self.epoch
            return True

        return False

    def __str__(self):
        rows = [
            "EarlyStopping",
            f"  Monitor: {self.monitor}.",
            f"  Patience: {self.patience}.",
        ]
        if self.stopped_epoch is not None:
            rows.append(f"  Stopped epoch: {self.stopped_epoch}.")
            rows.append(f"  Best score: {self.best_score:.6f}")

        return "\n".join(rows)
