"""
Benchmark problems for the hybrid PSO-GA algorithm.
"""

import numpy as np


# CONTINUOUS BENCHMARK - Ackley function

class Ackley:
    """
    Classic Ackley function - purely continuous, no discrete components.

    Highly multimodal: one global optimum at the origin surrounded by
    a large number of deceptive local optima.

    Easy version    
    """

    name = "Ackley"
    problem_type = "continuous"
    maximize = False
    n_continuous = 3
    discrete_options = [] # No discrete dimensions
    cont_lb      = np.array([-5.0] * 3) # 3 dimensions, each in [-5, 5]
    cont_ub      = np.array([ 5.0] * 3)
    optimum      = 0.0

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * np.pi
        n = len(x_cont)
        s1 = np.sum(x_cont ** 2)
        s2 = np.sum(np.cos(c * x_cont))
        return(
            -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e
        )






# DISCRETE BENCHMARK - 0/1 Knapsack problem

class Knapsack:
    """
    Classic 0/1 Knapsack problem - purely discrete, no continuous components.

    Select a subset of items to maximise total value without
    exceeding the weight capacity. Every decision is binary.

    """

    WEIGHTS       = np.array([0.4, 0.3, 0.6, 0.2, 0.5, 0.7, 0.1, 0.8])
    VALUES        = np.array([3.0, 2.0, 5.0, 1.5, 4.0, 6.0, 0.8, 7.0])
    CAPACITY      = 1.5

    name = "Knapsack"
    problem_type = "discrete"
    maximize = True
    n_continuous = 0
    discrete_options = [0, 1] * len(WEIGHTS) # Binary decisions for each item
    cont_lb = np.array([]) # No continuous dimensions
    cont_ub = np.array([]) # No continuous dimensions
    optimum = 18.8 

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        total_weight = float(np.dot(x_disc, self.WEIGHTS))
        total_value = float(np.dot(x_disc, self.VALUES))
        if total_weight > self.CAPACITY:
            return -1000.0 - (total_weight - self.CAPACITY) * 200.0 # Heavy penalty for exceeding capacity
        return total_value
   





# MIXED BENCHMARK - Neural Architecture Search (NAS) problem 
# Both x_cont and x_disc matter and interact

class NAS:
    """
    Toy Neural Architecture Search problem - a simple mixed optimization problem

    When designing a neural network you inherently face both types of
    decisions simultaneously:

    Continuous (handled by PSO):
        [0] learning rate  in [1e-4, 0.1]   — a real number
        [1] dropout rate   in [0.0,  0.5]

      Discrete (handled by GA):
        [0] n_layers    in {1, 2, 3, 4}
        [1] activation  in {relu=0, tanh=1, sigmoid=2}ù
        The fitness is a proxy for validation loss (no real training).
    """

    name = "NAS"
    problem_type = "mixed"
    maximize = False
    n_continuous = 2
    discrete_options = [4, 3] # 4 options for n_layers, 3 options for activation
    cont_lb = np.array([1e-4, 0.0]) # Learning rate, dropout rate
    cont_ub = np.array([0.1, 0.5])
    optimum = 0.0 # Hypothetical best validation loss


    LR_WEIGHT      = 80.0
    DROPOUT_WEIGHT = 0.8
    LAYER_WEIGHT   = 0.6
    ACT_PENALTY    = [0.0, 0.4, 1.0]    # relu best, sigmoid worst

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        lr, dropout  = float(x_cont[0]), float(x_cont[1])
        n_layers_idx = int(x_disc[0])
        act_idx      = int(x_disc[1])
        return (
            abs(lr - 0.01)          * self.LR_WEIGHT
            + abs(dropout - 0.1)    * self.DROPOUT_WEIGHT
            + abs(n_layers_idx - 1) * self.LAYER_WEIGHT
            + self.ACT_PENALTY[act_idx]
        )
    
BENCHMARKS = [Ackley(), Knapsack(), NAS()]