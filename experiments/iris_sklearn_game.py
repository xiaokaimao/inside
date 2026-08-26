"""Deterministic sklearn classification-accuracy data-valuation game."""

from __future__ import annotations

import numpy as np


class SklearnClassificationGame:
    """Accuracy utility for LR, linear SVM, or RBF SVM.

    Empty coalitions use the best constant-label accuracy on the performance
    set.  Single-class coalitions predict their sole observed class.  These
    rules make the game well-defined where sklearn classifiers cannot fit.
    """

    def __init__(
        self,
        *,
        X_valued: np.ndarray,
        y_valued: np.ndarray,
        X_performance: np.ndarray,
        y_performance: np.ndarray,
        num_classes: int,
        model: str,
        regularization: float = 1.0,
        game_seed: int = 2024,
        **_: object,
    ) -> None:
        self.X_valued = np.asarray(X_valued, dtype=np.float64)
        self.y_valued = np.asarray(y_valued, dtype=np.int64)
        self.X_performance = np.asarray(
            X_performance, dtype=np.float64
        )
        self.y_performance = np.asarray(
            y_performance, dtype=np.int64
        )
        self.num_players = len(self.y_valued)
        self.num_classes = int(num_classes)
        self.model_name = model
        self.regularization = float(regularization)
        self.game_seed = int(game_seed)
        class_counts = np.bincount(
            self.y_performance, minlength=self.num_classes
        )
        self.empty_score = float(class_counts.max() / len(self.y_performance))
        self.estimator = self._make_estimator()

    def _make_estimator(self):
        if self.model_name == "lr":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=self.regularization,
                solver="lbfgs",
                max_iter=200,
                random_state=self.game_seed,
            )
        if self.model_name == "linear_svm":
            from sklearn.svm import LinearSVC

            return LinearSVC(
                C=self.regularization,
                dual="auto",
                max_iter=5000,
                random_state=self.game_seed,
            )
        if self.model_name == "rbf_svm":
            from sklearn.svm import SVC

            return SVC(
                C=self.regularization,
                kernel="rbf",
                gamma="scale",
                probability=False,
                random_state=self.game_seed,
            )
        raise ValueError(
            "model must be 'lr', 'linear_svm', or 'rbf_svm'"
        )

    def evaluate(self, coalition: np.ndarray) -> float:
        mask = np.asarray(coalition, dtype=bool)[: self.num_players]
        if not np.any(mask):
            return self.empty_score
        labels = self.y_valued[mask]
        unique = np.unique(labels)
        if len(unique) == 1:
            return float(np.mean(self.y_performance == unique[0]))
        self.estimator.fit(self.X_valued[mask], labels)
        predictions = self.estimator.predict(self.X_performance)
        return float(np.mean(predictions == self.y_performance))


# Public backwards-compatible name used by the original Iris experiment.
IrisSklearnGame = SklearnClassificationGame
