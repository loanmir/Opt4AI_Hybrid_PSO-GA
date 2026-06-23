"""
    HybridParticle class which carries both a continuous part (updated by PSO) and a discrete part (updated by GA crossover/mutation).
"""

import numpy as np


class HybridParticle:
    """
        A single particle in the hybrid PSO-GA algorithm


        Attributes:
            - x_cont: -> current position in the continuous space
            - v_cont: -> current velocity in the continuous space
            - pb_cont: -> personal best position in the continuous space
            - x_disc: -> current position in the discrete space
            - pb_disc: -> personal best position in the discrete space
            - pb_fit: -> fitness value of the personal best position (same for both continuous and discrete parts)
            - fitness: -> current fitness value of the particle (same for both continuous and discrete parts)
    """


    def __init__(
            self, 
            n_continuous: int,
            discrete_options:list[int],
            cont_lb: np.ndarray,
            cont_ub: np.ndarray
    ):
        
        self.cont_lb = np.asarray(cont_lb, dtype=float)
        self.cont_ub = np.asarray(cont_ub, dtype=float)
        self.discrete_options = list(discrete_options)


        # PSO continuous part
        self.x_cont = np.random.uniform(self.cont_lb, self.cont_ub)
        self.v_cont = np.zeros(n_continuous)
        self.pb_cont = self.x_cont.copy()

        # GA discrete part
        self.x_disc = np.array(
            [np.random.randint(0, k) for k in discrete_options], dtype=int
        )
        self.pb_disc = self.x_disc.copy()


        # Fitness tracking
        self.fitness: float = np.inf
        self.pb_fit: float = np.inf






    """ Updating personal best if current position is better. 
        Return true if personal best is improved 
    """    
    def update_personal_best(self, maximize: bool) -> bool:
        is_better = self.fitness > self.pb_fit if maximize else self.fitness < self.pb_fit
        if is_better:
            self.pb_fit = self.fitness
            self.pb_cont = self.x_cont.copy()
            self.pb_disc = self.x_disc.copy()
            return True
        return False
    


    """ Returning the current solution as a tuple of (continuous part, discrete part)"""
    def solution(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_cont.copy(), self.x_disc.copy()
    

    """ Returning the personal best solution as a tuple of (continuous part, discrete part)"""
    def best_solution(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pb_cont.copy(), self.pb_disc.copy()
    



class GlobalBest:

    """ 
        Class that tracks the global best solution across the whole swarm.
        Internally it minimizes, the raw fitness values are negated on entry so the same
        "lower is better" logic can be applied everywhere in the algorithm.
        Having one single comparison everywhere 
    """

    def __init__(self, maximize: bool = False):
        self.maximize = maximize
        self.fitness: float = np.inf          # stored as internal minimisation value
        self.x_cont: np.ndarray | None = None
        self.x_disc: np.ndarray | None = None

    

    """Converting a raw fitness to the internal (minimisation) value."""
    def _to_internal(self, raw_fit: float) -> float:
        return -raw_fit if self.maximize else raw_fit
    


    """Fitness in natural units (un-negated for maximisation problems)."""
    @property
    def reported_fit(self) -> float:
        return -self.fitness if self.maximize else self.fitness
    


    """Updating global best from a particle's personal best. Returns True if improved."""
    def update(self, particle: HybridParticle) -> bool:
        internal = self._to_internal(particle.pb_fit)
        if internal < self.fitness:
            self.fitness = internal
            self.x_cont = particle.pb_cont.copy()
            self.x_disc = particle.pb_disc.copy()
            return True
        return False
    


    """Scanning the whole population and updating global best."""
    def update_from_all(self, population: list[HybridParticle]) -> None:
        for particle in population:
            self.update(particle)