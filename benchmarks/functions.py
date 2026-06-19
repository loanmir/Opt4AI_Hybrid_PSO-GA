"""
Benchmark problems for the hybrid PSO-GA algorithm.

CONTINUOUS BENCHMARK: Ackley function -> 3 dimensions, multimodal landscape, no discrete components.
DISCRETE BENCHMARK: 0/1 Knapsack problem -> 8 items, 2^8 = 256 combinations, binary decisions, no continuous components.
MIXED BENCHMARK: Toy Neural Architecture Search (NAS) problem -> 2 continuous dimensions (learning rate, dropout rate), 2 discrete dimensions (number of layers, activation function).
"""

import numpy as np


# CONTINUOUS BENCHMARK - Ackley function


#EASY VERSION OF ACKLEY FUNCTION 

class Ackley:
    """
    Classic Ackley function - purely continuous, no discrete components.

    Highly multimodal: one global optimum at the origin surrounded by
    a large number of deceptive local optima.ù

    Usually a stress-test for continuous optimizers

    Easy version
    Optimum: x=[0, 0, 0], f(x)=0    
    """

    name = "Ackley"
    problem_type = "continuous"
    maximize = False
    n_continuous = 3
    discrete_options = [] # No discrete dimensions
    cont_lb      = np.array([-5.0] * 3) # 3 dimensions, each in [-5, 5]
    cont_ub      = np.array([ 5.0] * 3)
    #optimum      = 0.0

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * np.pi
        n = len(x_cont)
        s1 = np.sum(x_cont ** 2)
        s2 = np.sum(np.cos(c * x_cont))
        return(
            -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e
        )


class HardAckley:
    """
        Hard Ackley - purely continuous, no discrete components.

        - 10 dimensions, standard bounds [-32.768, 32.768], global optimum at the origin with f(x)=0.
        - 10 dims landscape has more local optima than 3 dims 
        - Wider bounds mean random initialisation starts much farther
          from the optimum on average
    """

    name = "Hard Ackley"
    problem_type = "continuous"
    maximize = False
    n_continuous = 10
    discrete_options = [] # No discrete dimensions
    cont_lb      = np.array([-32.768] * 10) # 10 dimensions, each in [-32.768, 32.768]
    cont_ub      = np.array([32.768] * 10)
    #optimum      = 0.0 -> The optimum is at the origin, but we don't need to store it explicitly since it's known. -> LATER IMPLEMENT GAP BETWEEN BEST FOUND AND OPTIMUM 

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * np.pi
        n = len(x_cont)
        s1 = np.sum(x_cont ** 2)
        s2 = np.sum(np.cos(c * x_cont))
        return(
            -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e
        )




# DISCRETE BENCHMARK - 0/1 Knapsack problem


#EASY VERSION OF KNAPSACK PROBLEM

class Knapsack:
    """
    Classic 0/1 Knapsack problem - purely discrete, no continuous components.

    Select a subset of items to maximise total value without
    exceeding the weight capacity. Every decision is binary.

    Approximate optimum: select items 1, 2, 4, 5, 7 for a total value of 18.8
    """
    

    WEIGHTS       = np.array([0.4, 0.3, 0.6, 0.2, 0.5, 0.7, 0.1, 0.8])
    VALUES        = np.array([3.0, 2.0, 5.0, 1.5, 4.0, 6.0, 0.8, 7.0])
    CAPACITY      = 1.5

    name = "Knapsack"
    problem_type = "discrete"
    maximize = True
    n_continuous = 0
    discrete_options = [2] * len(WEIGHTS) # Binary decisions for each item
    cont_lb = np.array([]) # No continuous dimensions
    cont_ub = np.array([]) # No continuous dimensions
    #optimum = 13.0

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        total_weight = float(np.dot(x_disc, self.WEIGHTS))
        total_value = float(np.dot(x_disc, self.VALUES))
        if total_weight > self.CAPACITY:
            return -1000.0 - (total_weight - self.CAPACITY) * 200.0 # Heavy penalty for exceeding capacity
        return total_value
   


class HardKnapsack:
    """
        Hard Knapsack problem - purely discrete, no continuous components.

        - 25 items, 2 contraints
        - 2^25 = 33 million combinations make exhaustive search impossible
        - 2 simultaneous constraints (weight and budget) make it harder to find good solutions by chance (randomly)
        - GA crossover is essential: it preserves good partial item selections
    """

    _rng    = np.random.default_rng(42)    # fixed seed -> reproducible items
    WEIGHTS = np.round(_rng.uniform(0.1, 1.0,  25), 2)
    VALUES  = np.round(_rng.uniform(1.0, 10.0, 25), 1)
    COSTS   = np.round(_rng.uniform(0.5, 3.0,  25), 1) 
    CAPACITY = 6.0
    BUDGET   = 20.0

    name = "Hard Knapsack"
    problem_type = "discrete"
    maximize = True
    n_continuous = 0
    discrete_options = [2] * len(WEIGHTS) # Binary decisions for each item
    cont_lb = np.array([]) # No continuous dimensions
    cont_ub = np.array([]) # No continuous dimensions
    #optimum = 81.20  -> variable not used at the moment -> LATER IMPLEMENT GAP BETWEEN BEST FOUND AND OPTIMUM 

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        total_weight = float(np.dot(x_disc, self.WEIGHTS))
        total_value  = float(np.dot(x_disc, self.VALUES))
        total_cost   = float(np.dot(x_disc, self.COSTS))

        penalty = 0.0
        if total_weight > self.CAPACITY:
            penalty += (total_weight - self.CAPACITY) * 500.0
        if total_cost > self.BUDGET:
            penalty += (total_cost   - self.BUDGET)   * 200.0

        if penalty > 0:
            return -1000.0 - penalty
        return total_value




# MIXED BENCHMARK - Neural Architecture Search (NAS) problem 
# Both x_cont and x_disc matter and interact

# EASY VERSION OF NAS PROBLEM

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
        [1] activation  in {relu=0, tanh=1, sigmoid=2}

        The fitness is a proxy for validation loss (no real training).

        Optimum: lr=0.01, dropout=0.1, n_layers=2, activation=relu, f(x)=0
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


class HardNAS:
    """
        Hard Toy Neural Architecture Search problem - a more complex mixed optimization problem

        - 4 continuous + 5 discrete dimensions
        - 4 continuous dims with very different scales
        - 6*5*3*2*3 = 540 discrete combinations vs 4*3=12 in easy NAS
        - Genuine interactions between continuous and discrete choices

        - Continuous:
            [0] learning rate in [1e-5, 0.1] 
            [1] dropout rate in [0.0, 0.5]
            [2] weight decay in [1e-6, 1e-2]
            [3] log2(batch size) in [3.0, 8.0] (i.e. batch size 8 to 256)

        - Discrete:
            [0] n_layers in {1, 2, 3, 4, 5, 6}
            [1] activation in {relu, tanh, sigmoid, elu, selu}
            [2] optimizer in {sgd, adam, rmsprop}
            [3] skip_connections in {0=no, 1=yes}
            [4] normalization in {none, batchnorm, layernorm}
    """

    name = "Hard NAS"
    problem_type = "mixed"
    maximize = False
    n_continuous = 4
    discrete_options = [6, 5, 3, 2, 3] 
    cont_lb      = np.array([1e-5, 0.0,  1e-6, 3.0])
    cont_ub      = np.array([0.1,  0.5,  1e-2, 8.0])
    #optimum      = 0.0     ->    variable not used at the moment -> LATER IMPLEMENT GAP BETWEEN BEST FOUND AND OPTIMUM 


    OPT_LR = 0.001
    OPT_DO = 0.10
    OPT_WD = 1e-4
    OPT_BS = 5.0        # log2(32)

    ACT_PEN  = [0.0, 0.3, 0.8, 0.1, 0.15]   # relu,tanh,sigmoid,elu,selu
    OPT_PEN  = [0.0, 0.4, 0.2]              # adam,sgd,rmsprop
    NORM_PEN = [0.5, 0.0, 0.1]              # none,batchnorm,layernorm

    def fitness(self, x_cont: np.ndarray, x_disc: np.ndarray) -> float:
        lr, dropout, wd, bs_log = (float(v) for v in x_cont)
        n_layers_idx, act_idx, opt_idx, skip, norm_idx = (int(v) for v in x_disc)

        # --- Continuous cost ---
        cont_cost = (
            abs(lr      - self.OPT_LR) * 500.0
            + abs(dropout - self.OPT_DO) * 2.0
            + abs(wd      - self.OPT_WD) * 5000.0
            + abs(bs_log  - self.OPT_BS) * 0.3
        )

        # --- Discrete cost ---
        disc_cost = (
            abs(n_layers_idx - 2) * 0.8     # 3 layers = index 2
            + self.ACT_PEN[act_idx]
            + self.OPT_PEN[opt_idx]
            + self.NORM_PEN[norm_idx]
        )

        # --- Cross-type interaction: skip + batchnorm bonus ---
        # Awarded only when BOTH discrete choices are correct.
        # This reward can only be fully exploited when the continuous
        # params are also near-optimal — this is the cross-type
        # interaction that makes NAS genuinely mixed.
        interaction_bonus = 0.3 if (skip == 1 and norm_idx == 1) else 0.0

        # --- Continuous x discrete penalty: SGD + high lr is unstable ---
        sgd_lr_penalty = 0.5 if (opt_idx == 1 and lr > 0.01) else 0.0

        return cont_cost + disc_cost + sgd_lr_penalty - interaction_bonus


    
BENCHMARKS = [HardAckley(), HardKnapsack(), HardNAS()]