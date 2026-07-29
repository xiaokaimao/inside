"""Persistent, process-based evaluation of expensive coalition utilities."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from types import TracebackType
from typing import Any, Callable

import numpy as np

_WORKER_GAME: Any = None
_THREADPOOL_LIMITER: Any = None


def _initialize_game_worker(
    game_func: Callable[..., Any],
    game_args: dict[str, Any],
    worker_threads: int,
) -> None:
    """Construct one private game per worker and cap nested thread pools."""
    global _WORKER_GAME, _THREADPOOL_LIMITER
    thread_count = str(worker_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count

    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=worker_threads)
        _THREADPOOL_LIMITER.__enter__()
    except ImportError:
        _THREADPOOL_LIMITER = None

    try:
        import torch

        torch.set_num_threads(worker_threads)
        # PyTorch permits setting inter-op threads only before parallel work.
        try:
            torch.set_num_interop_threads(worker_threads)
        except RuntimeError:
            pass
    except ImportError:
        pass

    _WORKER_GAME = game_func(**game_args)


def _evaluate_worker(coalition: np.ndarray) -> float:
    if _WORKER_GAME is None:
        raise RuntimeError("utility worker was not initialized")
    return float(
        _WORKER_GAME.evaluate(np.asarray(coalition, dtype=bool))
    )


def _run_game_task(
    task: tuple[Callable[[Any, Any], Any], Any],
) -> Any:
    """Run one coarse task against the process-local game instance."""
    if _WORKER_GAME is None:
        raise RuntimeError("utility worker was not initialized")
    task_func, payload = task
    return task_func(_WORKER_GAME, payload)


class GameEvaluator:
    """Reuse a process pool whose workers each own a private game instance.

    The factory and its arguments must be pickleable when ``start_method`` is
    ``"spawn"``.  Callers must create the evaluator under an
    ``if __name__ == "__main__"`` guard on spawn-based platforms.
    """

    def __init__(
        self,
        game_func: Callable[..., Any],
        game_args: dict[str, Any],
        *,
        n_jobs: int = 1,
        chunksize: int = 16,
        start_method: str = "spawn",
        worker_threads: int = 1,
    ) -> None:
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive")
        if chunksize < 1:
            raise ValueError("chunksize must be positive")
        if worker_threads < 1:
            raise ValueError("worker_threads must be positive")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"unsupported multiprocessing start method: {start_method}"
            )
        self.game_func = game_func
        self.game_args = dict(game_args)
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.start_method = start_method
        self.worker_threads = worker_threads
        self._serial_game: Any = None
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "GameEvaluator":
        if self.n_jobs == 1:
            self._serial_game = self.game_func(**self.game_args)
        else:
            context = mp.get_context(self.start_method)
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_jobs,
                mp_context=context,
                initializer=_initialize_game_worker,
                initargs=(
                    self.game_func,
                    self.game_args,
                    self.worker_threads,
                ),
            )
        return self

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        rows = np.asarray(coalitions, dtype=bool)
        if rows.ndim != 2:
            raise ValueError("coalitions must be a two-dimensional array")
        if len(rows) == 0:
            return np.empty(0, dtype=np.float64)

        if self.n_jobs == 1:
            if self._serial_game is None:
                raise RuntimeError("GameEvaluator must be used as a context")
            return np.asarray(
                [
                    self._serial_game.evaluate(row.copy())
                    for row in rows
                ],
                dtype=np.float64,
            )

        if self._executor is None:
            raise RuntimeError("GameEvaluator must be used as a context")
        values = self._executor.map(
            _evaluate_worker,
            rows,
            chunksize=self.chunksize,
        )
        return np.fromiter(values, dtype=np.float64, count=len(rows))

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]:
        """Run coarse, game-aware tasks on the persistent workers.

        This is intended for algorithms such as permutation Monte Carlo where
        constructing and evaluating a complete trajectory inside a worker is
        much cheaper than sending every coalition through inter-process
        communication.  Keep ``payloads`` to a bounded outer batch when using
        Python versions whose executor ``map`` eagerly submits all work.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be positive")
        if not payloads:
            return []
        if self.n_jobs == 1:
            if self._serial_game is None:
                raise RuntimeError("GameEvaluator must be used as a context")
            return [
                task_func(self._serial_game, payload)
                for payload in payloads
            ]
        if self._executor is None:
            raise RuntimeError("GameEvaluator must be used as a context")
        tasks = [(task_func, payload) for payload in payloads]
        return list(
            self._executor.map(
                _run_game_task,
                tasks,
                chunksize=chunksize,
            )
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self._serial_game = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def evaluate_game_coalitions(
    game_func: Callable[..., Any],
    game_args: dict[str, Any],
    coalitions: np.ndarray,
    *,
    n_jobs: int = 1,
    chunksize: int = 16,
    start_method: str = "spawn",
    worker_threads: int = 1,
) -> np.ndarray:
    """One-shot convenience wrapper around :class:`GameEvaluator`."""
    with GameEvaluator(
        game_func,
        game_args,
        n_jobs=n_jobs,
        chunksize=chunksize,
        start_method=start_method,
        worker_threads=worker_threads,
    ) as evaluator:
        return evaluator.evaluate(coalitions)
