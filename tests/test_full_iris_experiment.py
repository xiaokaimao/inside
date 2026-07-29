from __future__ import annotations

import itertools
import unittest

import numpy as np

from experiments.iris_data_valuation import ground_truth_summary
from experiments.iris_full_train_data_valuation import (
    _antithetic_pair_task,
    _permutation_path_contribution,
    summarize_pair_moments,
)
from experiments.iris_game import load_iris_train_test_split


class TableGame:
    def __init__(self, table: np.ndarray) -> None:
        self.table = np.asarray(table, dtype=np.float64)

    def evaluate(self, coalition: np.ndarray) -> float:
        mask = 0
        for player in np.flatnonzero(coalition):
            mask |= 1 << int(player)
        return float(self.table[mask])


class FullIrisExperimentTests(unittest.TestCase):
    def test_full_split_uses_all_training_rows_without_leakage(self) -> None:
        game_args, metadata = load_iris_train_test_split(
            test_size=0.2,
            dataset_seed=2024,
        )
        self.assertEqual(len(game_args["y_valued"]), 120)
        self.assertEqual(len(game_args["y_performance"]), 30)
        self.assertEqual(metadata["train_class_counts"], [40, 40, 40])
        self.assertEqual(metadata["test_class_counts"], [10, 10, 10])
        self.assertEqual(
            set(metadata["train_original_indices"]).intersection(
                metadata["test_original_indices"]
            ),
            set(),
        )
        np.testing.assert_allclose(
            np.asarray(game_args["X_valued"]).mean(axis=0),
            0.0,
            atol=5e-15,
        )

    def test_boundary_reused_path_matches_direct_prefix_path(self) -> None:
        rng = np.random.default_rng(51)
        for num_players in range(4, 8):
            table = rng.normal(size=1 << num_players)
            game = TableGame(table)
            singletons = np.asarray(
                [table[1 << player] for player in range(num_players)]
            )
            full_mask = (1 << num_players) - 1
            leave_one_out = np.asarray(
                [
                    table[full_mask ^ (1 << player)]
                    for player in range(num_players)
                ]
            )
            for permutation_tuple in itertools.islice(
                itertools.permutations(range(num_players)),
                0,
                30,
            ):
                permutation = np.asarray(permutation_tuple)
                expected = np.zeros(num_players, dtype=np.float64)
                previous = table[0]
                coalition = np.zeros(num_players, dtype=bool)
                for player in permutation:
                    coalition[player] = True
                    current = game.evaluate(coalition)
                    expected[player] = current - previous
                    previous = current
                actual = _permutation_path_contribution(
                    game,
                    permutation,
                    empty_utility=float(table[0]),
                    full_utility=float(table[-1]),
                    singletons=singletons,
                    leave_one_out=leave_one_out,
                )
                np.testing.assert_allclose(actual, expected, atol=0.0)

    def test_antithetic_task_returns_pair_averages_in_order(self) -> None:
        num_players = 6
        rng = np.random.default_rng(19)
        table = rng.normal(size=1 << num_players)
        game = TableGame(table)
        forward = np.stack(
            [rng.permutation(num_players) for _ in range(7)]
        )
        full_mask = (1 << num_players) - 1
        singletons = np.asarray(
            [table[1 << player] for player in range(num_players)]
        )
        leave_one_out = np.asarray(
            [
                table[full_mask ^ (1 << player)]
                for player in range(num_players)
            ]
        )
        actual = _antithetic_pair_task(
            game,
            (
                forward,
                float(table[0]),
                float(table[-1]),
                singletons,
                leave_one_out,
            ),
        )
        expected = []
        for permutation in forward:
            first = _permutation_path_contribution(
                game,
                permutation,
                empty_utility=float(table[0]),
                full_utility=float(table[-1]),
                singletons=singletons,
                leave_one_out=leave_one_out,
            )
            second = _permutation_path_contribution(
                game,
                permutation[::-1],
                empty_utility=float(table[0]),
                full_utility=float(table[-1]),
                singletons=singletons,
                leave_one_out=leave_one_out,
            )
            expected.append((first + second) / 2)
        np.testing.assert_allclose(actual, expected, atol=0.0)

    def test_streaming_moment_summary_matches_full_summary(self) -> None:
        rng = np.random.default_rng(82)
        pairs = rng.normal(size=(100, 9))
        expected = ground_truth_summary(pairs)
        actual = summarize_pair_moments(
            count=len(pairs),
            total=pairs.sum(axis=0),
            total_squares=np.square(pairs).sum(axis=0),
            first_half_count=50,
            first_half_total=pairs[:50].sum(axis=0),
            second_half_count=50,
            second_half_total=pairs[50:].sum(axis=0),
        )
        for key in (
            "values",
            "standard_errors",
            "simultaneous_half_widths",
        ):
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=2e-14, atol=2e-14
            )
        for key in (
            "rmse_standard_error",
            "max_simultaneous_half_width",
            "half_split_rmse",
        ):
            self.assertAlmostEqual(actual[key], expected[key], places=14)


if __name__ == "__main__":
    unittest.main()
