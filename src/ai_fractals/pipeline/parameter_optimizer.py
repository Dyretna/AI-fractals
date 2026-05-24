"""
Parameter space optimization for finding interesting fractals.
Implements Bayesian optimization and evolutionary strategies.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ParameterSample:
    params: Dict[str, float]
    score: float
    fractal_dimension: Optional[float] = None


class BayesianOptimizer:
    """
    Bayesian optimization for parameter search.
    Balances exploration vs exploitation using Upper Confidence Bound (UCB).
    """

    def __init__(self, param_ranges: Dict, exploration_weight: float = 2.0):
        self.param_ranges = param_ranges
        self.exploration_weight = exploration_weight
        self.samples: List[ParameterSample] = []
        self.best_score = 0.0
        self.best_params = None

    def suggest_parameters(self, strategy: str = "ucb") -> Dict[str, float]:
        if len(self.samples) < 10:
            return self._random_sample()

        if strategy == "ucb":
            return self._ucb_sample()
        elif strategy == "exploit":
            return self._exploit_best()
        else:
            return self._random_sample()

    def _random_sample(self) -> Dict[str, float]:
        params = {}
        for key, range_obj in self.param_ranges.items():
            if hasattr(range_obj, "sample"):
                val = range_obj.sample()
            else:
                val = np.random.uniform(range_obj["min"], range_obj["max"])
            # Ensure integers for dimension parameters
            params[key] = (
                int(max(1, val)) if key in ["size", "max_iter"] else float(val)
            )
        return params

    def _ucb_sample(self) -> Dict[str, float]:
        if not self.samples:
            return self._random_sample()

        top_samples = sorted(self.samples, key=lambda x: x.score, reverse=True)[:5]
        base_sample = np.random.choice(top_samples)

        params = {}
        for key, value in base_sample.params.items():
            if key in ["size", "max_iter"]:
                noise_scale = 0.1
                noise = np.random.normal(0, noise_scale * abs(value + 1e-6))
                params[key] = int(max(1, value + noise))
            else:
                noise_scale = 0.2
                noise = np.random.normal(0, noise_scale * abs(value + 1e-6))
                params[key] = float(value + noise)
        if not self.best_params:
            return self._random_sample()

        # Small perturbation around best
        params = {}
        for key, value in self.best_params.items():
            noise = np.random.normal(0, 0.05 * abs(value + 1e-6))
            new_val = value + noise
            # Ensure integers for dimension parameters
            params[key] = (
                int(max(1, new_val)) if key in ["size", "max_iter"] else float(new_val)
            )

        return params

    def update(
        self,
        params: Dict[str, float],
        score: float,
        fractal_dimension: Optional[float] = None,
    ):
        sample = ParameterSample(params, score, fractal_dimension)
        self.samples.append(sample)

        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

        # Keep only recent samples
        if len(self.samples) > 1000:
            self.samples = self.samples[-1000:]


class EvolutionaryOptimizer:
    """
    Evolutionary algorithm for parameter optimization.
    Uses tournament selection and mutation.
    """

    def __init__(self, param_ranges: Dict, population_size: int = 20):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.population: List[ParameterSample] = []
        self.generation = 0

    def suggest_parameters(self) -> Dict[str, float]:
        if len(self.population) < self.population_size:
            return self._random_sample()

        # Tournament selection
        parent1 = self._tournament_select()
        parent2 = self._tournament_select()

        # Crossover
        child_params = self._crossover(parent1.params, parent2.params)

        # Mutation
        child_params = self._mutate(child_params)

        return child_params

    def _random_sample(self) -> Dict[str, float]:
        params = {}
        for key, range_obj in self.param_ranges.items():
            if hasattr(range_obj, "sample"):
                val = range_obj.sample()
            else:
                val = np.random.uniform(range_obj["min"], range_obj["max"])
            # Ensure integers for dimension parameters
            params[key] = (
                int(max(1, val)) if key in ["size", "max_iter"] else float(val)
            )
        return params

    def _tournament_select(self, tournament_size: int = 3) -> ParameterSample:
        candidates = np.random.choice(
            self.population, size=tournament_size, replace=False
        )
        return max(candidates, key=lambda x: x.score)

    def _crossover(self, params1: Dict, params2: Dict) -> Dict[str, float]:
        child = {}
        for key in params1.keys():
            val = params1[key] if np.random.random() < 0.5 else params2[key]
            if key in ["size", "max_iter"]:
                child[key] = int(val)
            else:
                child[key] = float(val)
        return child

    def _mutate(
        self, params: Dict[str, float], mutation_rate: float = 0.3
    ) -> Dict[str, float]:
        mutated = {}
        for key, value in params.items():
            if np.random.random() < mutation_rate:
                if key in ["size", "max_iter"]:
                    noise_scale = 0.2
                else:
                    noise_scale = 0.3
                noise = np.random.normal(0, noise_scale * abs(value + 1e-6))
                new_val = value + noise
                mutated[key] = (
                    int(max(1, new_val))
                    if key in ["size", "max_iter"]
                    else float(new_val)
                )
            else:
                mutated[key] = (
                    int(value) if key in ["size", "max_iter"] else float(value)
                )
        return mutated

    def update(
        self,
        params: Dict[str, float],
        score: float,
        fractal_dimension: Optional[float] = None,
    ):
        sample = ParameterSample(params, score, fractal_dimension)
        self.population.append(sample)

        # Keep best individuals
        if len(self.population) > self.population_size:
            self.population = sorted(
                self.population, key=lambda x: x.score, reverse=True
            )
            self.population = self.population[: self.population_size]

        self.generation += 1


class AdaptiveParameterSearch:
    """
    Adaptive search combining multiple strategies.
    Switches between exploration and exploitation based on success rate.
    """

    def __init__(self, param_ranges: Dict, window_size: int = 50):
        self.param_ranges = param_ranges
        self.window_size = window_size
        self.recent_scores = deque(maxlen=window_size)
        self.bayesian_opt = BayesianOptimizer(param_ranges)
        self.evolutionary_opt = EvolutionaryOptimizer(param_ranges)
        self.mode = "explore"

    def suggest_parameters(self) -> Dict[str, float]:
        # Decide strategy based on recent performance
        if len(self.recent_scores) < self.window_size:
            self.mode = "explore"
        else:
            avg_score = np.mean(list(self.recent_scores))
            if avg_score < 0.5:
                self.mode = "explore"
            elif avg_score < 0.7:
                self.mode = "mixed"
            else:
                self.mode = "exploit"

        # Select optimizer
        if self.mode == "explore":
            # 70% Bayesian (UCB), 30% evolutionary
            if np.random.random() < 0.7:
                return self.bayesian_opt.suggest_parameters("ucb")
            else:
                return self.evolutionary_opt.suggest_parameters()

        elif self.mode == "mixed":
            # 50% Bayesian, 50% evolutionary
            if np.random.random() < 0.5:
                return self.bayesian_opt.suggest_parameters("ucb")
            else:
                return self.evolutionary_opt.suggest_parameters()

        else:  # exploit
            # Focus on best regions
            return self.bayesian_opt.suggest_parameters("exploit")

    def update(
        self,
        params: Dict[str, float],
        score: float,
        fractal_dimension: Optional[float] = None,
    ):
        self.recent_scores.append(score)
        self.bayesian_opt.update(params, score, fractal_dimension)
        self.evolutionary_opt.update(params, score, fractal_dimension)

    def get_status(self) -> Dict:
        return {
            "mode": self.mode,
            "recent_avg_score": np.mean(list(self.recent_scores))
            if self.recent_scores
            else 0.0,
            "best_score": self.bayesian_opt.best_score,
            "population_size": len(self.evolutionary_opt.population),
            "generation": self.evolutionary_opt.generation,
        }
