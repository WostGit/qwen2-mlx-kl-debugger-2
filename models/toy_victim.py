import numpy as np


class ToyVictimModel:
    """Deterministic toy victim producing dense probabilities over classes."""

    def __init__(self, n_classes: int = 16, seed: int = 0):
        self.n_classes = n_classes
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1, size=(8, n_classes))
        self.b = rng.normal(0, 0.1, size=(n_classes,))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    def featurize(self, query_id: int) -> np.ndarray:
        feats = np.array([
            np.sin(query_id),
            np.cos(query_id),
            query_id % 2,
            query_id % 3,
            query_id % 5,
            query_id / 10.0,
            (query_id ** 2) % 7,
            1.0,
        ], dtype=np.float64)
        return feats

    def probs(self, query_id: int) -> np.ndarray:
        logits = self.featurize(query_id) @ self.W + self.b
        p = self.softmax(logits)
        return p.astype(np.float64)

    def argmax(self, query_id: int) -> int:
        return int(np.argmax(self.probs(query_id)))

    def topk(self, query_id: int, k: int = 3):
        p = self.probs(query_id)
        idx = np.argsort(p)[::-1][:k]
        return idx.tolist(), p[idx].tolist()
