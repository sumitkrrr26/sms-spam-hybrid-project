from __future__ import annotations

import math
from typing import Callable

import numpy as np


class GSOWOA:
    """
    Compact research implementation of a Golden-Section + Whale Optimization
    search over continuous hyperparameters.

    Objective must return a score to MAXIMIZE (here, validation spam F1).
    This implementation is an explicit project implementation, not a claim
    that every detail matches another paper's exact GSOWOA variant.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        population_size: int = 3,
        iterations: int = 1,
        golden_steps: int = 1,
        seed: int = 42,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if iterations < 0 or golden_steps < 0:
            raise ValueError("iterations and golden_steps must be non-negative.")
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.population_size = population_size
        self.iterations = iterations
        self.golden_steps = golden_steps
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self, objective: Callable[[np.ndarray], float]):
        population = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.population_size, len(self.bounds))
        )
        scores = []
        for index, position in enumerate(population):
            score = float(objective(position.copy()))
            scores.append(score)
            self.history.append({
                "stage": "initial", "candidate": index,
                "score": score, "position": position.tolist(),
            })

        scores = np.asarray(scores, dtype=np.float64)
        best_index = int(np.argmax(scores))
        best_position = population[best_index].copy()
        best_score = float(scores[best_index])

        # Whale Optimization Algorithm population updates.
        for iteration in range(self.iterations):
            a = 2.0 - 2.0 * (iteration / max(1, self.iterations))
            next_population = []
            for i, current in enumerate(population):
                r1, r2 = self.rng.random(), self.rng.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = self.rng.random()
                if p < 0.5:
                    if abs(A) < 1:
                        distance = np.abs(C * best_position - current)
                        candidate = best_position - A * distance
                    else:
                        random_whale = population[self.rng.integers(0, len(population))]
                        distance = np.abs(C * random_whale - current)
                        candidate = random_whale - A * distance
                else:
                    spiral_l = self.rng.uniform(-1.0, 1.0)
                    distance = np.abs(best_position - current)
                    candidate = (
                        distance * math.exp(1.0 * spiral_l)
                        * math.cos(2.0 * math.pi * spiral_l) + best_position
                    )

                candidate = self._clip(candidate)
                score = float(objective(candidate.copy()))
                self.history.append({
                    "stage": f"whale_iteration_{iteration + 1}",
                    "candidate": i,
                    "score": score,
                    "position": candidate.tolist(),
                })
                next_population.append(candidate)
                if score > best_score:
                    best_position, best_score = candidate.copy(), score

            population = np.asarray(next_population, dtype=np.float64)
            scores = np.asarray([
                item["score"] for item in self.history[-self.population_size:]
            ], dtype=np.float64)
            # Elitism: replace the worst whale with the best-so-far position.
            worst = int(np.argmin(scores))
            population[worst] = best_position.copy()

        # Golden-section local search on log10(learning_rate) dimension only.
        if self.golden_steps > 0:
            lo, hi = self.bounds[0]
            center = float(best_position[0])
            left = max(lo, center - 0.5)
            right = min(hi, center + 0.5)
            phi = (math.sqrt(5.0) - 1.0) / 2.0
            c = right - phi * (right - left)
            d = left + phi * (right - left)
            xc, xd = best_position.copy(), best_position.copy()
            xc[0], xd[0] = c, d
            fc, fd = float(objective(xc.copy())), float(objective(xd.copy()))
            self.history.append({
                "stage": "golden_section", "candidate": "c",
                "score": fc, "position": xc.tolist(),
            })
            self.history.append({
                "stage": "golden_section", "candidate": "d",
                "score": fd, "position": xd.tolist(),
            })
            for score, position in ((fc, xc), (fd, xd)):
                if score > best_score:
                    best_position, best_score = position.copy(), score

        return best_position, best_score, self.history
