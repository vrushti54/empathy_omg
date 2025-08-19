class EarlyStopper:
    def __init__(self, patience=5, mode="max", min_delta=1e-6):
        """
        mode='max'  -> metric must INCREASE by min_delta to count as improvement (e.g., CCC)
        mode='min'  -> metric must DECREASE by min_delta to count as improvement (e.g., loss)
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False  # don't stop
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
