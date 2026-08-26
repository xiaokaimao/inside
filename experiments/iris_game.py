"""Official-style deterministic Iris data-valuation game."""

from __future__ import annotations

from typing import Any

import numpy as np

from .sklearn_data import load_sklearn_train_test_split


def load_iris_train_test_split(
    *,
    test_size: float = 0.2,
    dataset_seed: int = 2024,
) -> tuple[dict[str, np.ndarray | int | float], dict[str, Any]]:
    """Use every training observation as a data-valuation player.

    The split is stratified, and feature standardization is fitted on the
    training partition only.  With the standard Iris data and
    ``test_size=0.2``, this produces 120 valued players and 30 test examples.
    """
    return load_sklearn_train_test_split(
        "iris",
        test_size=test_size,
        dataset_seed=dataset_seed,
    )


def load_official_iris_split(
    *,
    n_valued: int = 24,
    n_performance: int = 24,
    dataset_seed: int = 2024,
) -> tuple[dict[str, np.ndarray | int | float], dict[str, Any]]:
    """Reproduce the upstream OFA balanced Iris split without OpenML."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = np.asarray(iris.data, dtype=np.float64)
    target = np.asarray(iris.target, dtype=np.int64)
    num_classes = len(np.unique(target))

    split_rng = np.random.RandomState(dataset_seed)
    permutation = split_rng.permutation(len(target))
    num_train = int(np.round(len(target) * 0.8))
    train_positions = permutation[:num_train]
    data_train = data[train_positions]
    label_train = target[train_positions]
    data_mean = data_train.mean(axis=0, keepdims=True)
    data_std = data_train.std(axis=0, keepdims=True)
    data_train = (data_train - data_mean) / data_std

    positions_by_class = {
        label: np.flatnonzero(label_train == label)
        for label in range(num_classes)
    }
    minimum_class_count = min(map(len, positions_by_class.values()))
    if n_valued + n_performance > minimum_class_count * num_classes:
        raise ValueError("requested split exceeds balanced training data")

    class_offsets = np.zeros(num_classes, dtype=np.int64)
    valued_positions = np.empty(n_valued, dtype=np.int64)
    performance_positions = np.empty(n_performance, dtype=np.int64)
    for cursor in range(n_valued + n_performance):
        label = cursor % num_classes
        source = positions_by_class[label]
        selected = source[class_offsets[label]]
        if cursor < n_valued:
            valued_positions[cursor] = selected
        else:
            performance_positions[cursor - n_valued] = selected
        class_offsets[label] += 1

    shuffle_rng = np.random.RandomState(dataset_seed)
    shuffle_rng.shuffle(valued_positions)
    shuffle_rng.shuffle(performance_positions)
    game_args: dict[str, np.ndarray | int | float] = {
        "X_valued": data_train[valued_positions],
        "y_valued": label_train[valued_positions],
        "X_performance": data_train[performance_positions],
        "y_performance": label_train[performance_positions],
        "num_classes": num_classes,
        "learning_rate": 1.0,
        "game_seed": 2024,
    }
    metadata: dict[str, Any] = {
        "dataset": "iris",
        "dataset_seed": dataset_seed,
        "train_positions_original": train_positions.tolist(),
        "valued_positions_within_train": valued_positions.tolist(),
        "performance_positions_within_train": performance_positions.tolist(),
        "valued_original_indices": train_positions[
            valued_positions
        ].tolist(),
        "performance_original_indices": train_positions[
            performance_positions
        ].tolist(),
        "valued_labels": label_train[valued_positions].tolist(),
        "performance_labels": label_train[
            performance_positions
        ].tolist(),
        "standardization_mean": data_mean.ravel().tolist(),
        "standardization_std": data_std.ravel().tolist(),
    }
    return game_args, metadata


class IrisLogisticGame:
    """Match upstream ``gameTraining`` for the Iris logistic experiment."""

    def __init__(
        self,
        *,
        X_valued: np.ndarray,
        y_valued: np.ndarray,
        X_performance: np.ndarray,
        y_performance: np.ndarray,
        num_classes: int,
        learning_rate: float,
        game_seed: int,
    ) -> None:
        import torch

        self.torch = torch
        self.X_valued = torch.as_tensor(X_valued, dtype=torch.float64)
        self.y_valued = torch.as_tensor(y_valued, dtype=torch.int64)
        self.X_performance = torch.as_tensor(
            X_performance, dtype=torch.float64
        )
        self.y_performance = torch.as_tensor(
            y_performance, dtype=torch.int64
        )
        self.num_players = len(self.y_valued)
        self.num_classes = int(num_classes)
        self.half_num_classes = self.num_classes // 2
        self.game_seed = int(game_seed)
        self.model = torch.nn.Linear(
            self.X_performance.shape[1], self.num_classes
        ).double()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=float(learning_rate)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def _reset_model(self) -> None:
        torch = self.torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.game_seed)
            self.model.reset_parameters()

    def evaluate(self, coalition: np.ndarray) -> float:
        mask = np.asarray(coalition, dtype=bool)[: self.num_players]
        torch = self.torch
        mask_tensor = torch.from_numpy(mask)
        X = self.X_valued[mask_tensor]
        y = self.y_valued[mask_tensor]
        order_seed = int(
            (X > 0).sum().item()
            + (y < self.half_num_classes).sum().item()
            + self.game_seed
        )
        order = np.random.RandomState(order_seed).permutation(len(y))
        X = X[order]
        y = y[order]

        self._reset_model()
        for datum, label in zip(X, y):
            logits = self.model(datum)
            loss = self.criterion(logits, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_performance)
            score = -self.criterion(
                logits, self.y_performance
            ).item()
        self.model.train()
        return float(score)
