"""
Three optimizers that share same interface, in order to have a better comparison between them on the same benchmark.

    run_hybrid -> run full Hybrid PSO-GA
    run_pure_pso -> PSO applied to both continuous and discrete dimensions
    run_pure_ga -> GA applied to both continuous and discrete dimensions

All return an AlgorithmResult dataclass 

"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field  # Python decorator to automatically generate special methods like __init__() and __repr__() for classes that primarily store data.
from typing import Callable
from core.particle import HybridParticle, GlobalBest
from core.operators import pso_update, ga_update, tournament_selection


FitnessFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class AlgorithmResult:
    name: str
    best_continuous: np.ndarray
    best_discrete: np.ndarray
    best_fitness: float
    history: list[float] 
    n_evaluations: int
    maximize: bool = False 


def _make_population(
        n_particles: int,
        n_continuous: int,
        discrete_options: list[int],
        cont_lb: np.ndarray,
        cont_ub: np.ndarray,
        fitness_fn: FitnessFn,
        maximize: bool = False,
) -> tuple[list[HybridParticle], GlobalBest]:
    """ Initialise population and evaluation of initial fitness """
    pop = [
        HybridParticle(
            n_continuous=n_continuous,
            discrete_options=discrete_options,
            cont_lb=cont_lb,
            cont_ub=cont_ub,
        )
        for _ in range(n_particles)
    ]
    global_best = GlobalBest(maximize=maximize)
    for particle in pop:
        particle.fitness = fitness_fn(particle.x_cont, particle.x_disc)
        particle.pb_fit = -np.inf if maximize else np.inf # For maximisation, the personal best starts at -inf;
        particle.update_personal_best(maximize=maximize)
        global_best.update(particle)
    return pop, global_best




# Hybrid PSO-GA ---------------------------------------------------

def run_hybrid(
        fitner_fn: FitnessFn,
        n_continuous: int,
        discrete_options: list[int],
        cont_lb: np.ndarray,
        cont_ub: np.ndarray,
        n_particles: int = 30,
        # *, -> not necessarly needed -> Used for defing that everything after the * can only be passed by the name (not just value, so keyword=value)
        max_iters: int = 300,
        w: float = 0.7, # INERTIA WEIGHT for PSO -> Controls how much of the particle's previous velocity it carries forward
        c1: float = 1.5,
        c2: float = 1.5,
        p_cross: float = 0.7,
        p_mut: float = 0.1,
        ga_every: int = 1, # -> GA xover and mutation on discrete dims EVERY N ITERS
        w_decay: bool = True, # -> INERTIA WEIGHT DECAY -> Discreases linearly the w value -> It balances exploration early and exploitation late in the run
        maximize: bool = False
) -> AlgorithmResult: 
    
    """ Hybrid PSO-GA main loop

        PSO updates the continuous dimensions every iteration
        GA updates the discrete dimensions every 'ga_every' iterations
    """

    cont_lb = np.asarray(cont_lb, dtype=float)
    cont_ub = np.asarray(cont_ub, dtype=float)
    pop, global_best = _make_population(
        n_particles=n_particles,
        n_continuous=n_continuous,
        discrete_options=discrete_options,
        cont_lb=cont_lb,
        cont_ub=cont_ub,
        fitness_fn=fitner_fn,
        maximize=maximize
    )

    history = [global_best.reported_fit]
    n_evals = n_particles 

    w_start = w
    w_end = 0.4 # Final inertia weight after decay

    for iter in range(1, max_iters + 1):
        w_current = w_start - (w_start - w_end) * iter / max_iters if w_decay else w  # value of inertia weight at current iteration 

        for particle in pop:

            # PSO update for continuous dimensions
            pso_update(particle, global_best.x_cont, w_current, c1, c2)

            # GA update for discrete dimensions every 'ga_every' iterations
            if iter % ga_every == 0:
                # Select a partner for crossover using tournament selection
                partner = tournament_selection(pop, k=3)
                ga_update(particle, partner, fitner_fn, p_cross, p_mut, maximize=maximize)    # making the fitness evaluation inside the ga_update function also
                #ga_update(particle, partner, p_cross, p_mut, maximize=maximize)

            # Fitness evaluation
            particle.fitness = fitner_fn(particle.x_cont, particle.x_disc)     # -> Doing a double fitness evaluation 
            n_evals += 1

            # Updating personal best and global best
            particle.update_personal_best(maximize=maximize)
            global_best.update(particle)

        history.append(global_best.reported_fit)

    return AlgorithmResult(
        name = "Hybrid PSO-GA",
        best_continuous=global_best.x_cont,
        best_discrete=global_best.x_disc,
        best_fitness=global_best.reported_fit,
        history=history,
        n_evaluations=n_evals,
        maximize=maximize
    )







# Pure PSO ---------------------------------------------------
# (applies PSO velocity to continuous dims; rounds velocity for discrete dims)

def run_pure_pso(
            fitner_fn: FitnessFn,
            n_continuous: int,
            discrete_options: list[int],
            cont_lb: np.ndarray,
            cont_ub: np.ndarray,
            # *,
            n_particles: int = 30,
            max_iters: int = 300,
            w: float = 0.7, # INERTIA WEIGHT for PSO -> Controls how much of the particle's previous velocity it carries forward
            c1: float = 1.5,
            c2: float = 1.5,
            w_decay: bool = True, # -> INERTIA WEIGHT DECAY -> Discreases linearly the w value -> It balances exploration early and exploitation late in the run
            maximize: bool = False
) -> AlgorithmResult:
    """ Pure PSO main loop
        PSO updates continuous dimensions with standard velocity update
        PSO updates discrete dimensions by rounding the velocity to the nearest valid integer option
    """
    cont_lb = np.asarray(cont_lb, dtype=float)
    cont_ub = np.asarray(cont_ub, dtype=float)

    pop, global_best = _make_population(
        n_particles=n_particles,
        n_continuous=n_continuous,
        discrete_options=discrete_options,
        cont_lb=cont_lb,
        cont_ub=cont_ub,
        fitness_fn=fitner_fn,
        maximize=maximize
    )

    disc_v = {id(particle): np.zeros(len(discrete_options)) for particle in pop} # Separate velocity for discrete dimensions

    history = [global_best.reported_fit]
    n_evals = n_particles 

    w_start = w
    w_end = 0.4 

    for iter in range(1, max_iters + 1):
        w_current = w_start - (w_start - w_end) * iter / max_iters if w_decay else w  # value of inertia weight at current iteration 

        for particle in pop:

            # PSO update for continuous dimensions
            pso_update(particle, global_best.x_cont, w_current, c1, c2)

            # PSO update for discrete dimensions by rounding velocity to nearest valid option
            if len(discrete_options) > 0:
                    dv = disc_v[id(particle)] # current discrete velocity of particle from dictionary
                    n_d = len(particle.x_disc) # Number of discrete dimensions
                    r1 = np.random.rand(n_d) # Random numbers for personal component
                    r2 = np.random.rand(n_d) # Random numbers for global component
                    # Velocity update using PSO formula
                    dv = w_current * dv + c1 * r1 * (particle.pb_disc - particle.x_disc) + c2 * r2 * (global_best.x_disc - particle.x_disc)
                    disc_v[id(particle)] = dv
                    new_disc = np.round(particle.x_disc + dv).astype(int) # Adding velocity to current position and rounding to nearest integer
                    particle.x_disc = np.clip(new_disc, 0, np.array(discrete_options) - 1) # Ensure discrete values stay within valid range, within the bounds

            # Fitness evaluation
            particle.fitness = fitner_fn(particle.x_cont, particle.x_disc)
            n_evals += 1

            # Updating personal best and global best
            particle.update_personal_best(maximize=maximize)
            global_best.update(particle)

        history.append(global_best.reported_fit)

    return AlgorithmResult(
        name = "Pure PSO",
        best_continuous=global_best.x_cont,
        best_discrete=global_best.x_disc,
        best_fitness=global_best.reported_fit,
        history=history,
        n_evaluations=n_evals,
        maximize=maximize
    )



# Pure GA ----------------------------------------------------
# (applies crossover + mutation to both continuous and discrete dims)
def run_pure_ga(
        fintess_fn: FitnessFn,
        n_continuous: int,
        discrete_options: list[int],
        cont_lb: np.ndarray,
        cont_ub: np.ndarray,
        # *,
        n_particles: int = 30,
        max_iters: int = 300,
        p_cross: float = 0.7,
        p_mut: float = 0.05,
        maximize: bool = False
) -> AlgorithmResult:
    """ Pure GA main loop
        Continuous dimensions: Arithmetic xover + Gaussian mutation
        Discrete dimensions: Uniform xover + Random reset mutation
        Selection is tournament selection (k=3)
    """
    cont_lb = np.asarray(cont_lb, dtype=float)
    cont_ub = np.asarray(cont_ub, dtype=float)

    pop, global_best = _make_population(
        n_particles=n_particles,
        n_continuous=n_continuous,
        discrete_options=discrete_options,
        cont_lb=cont_lb,
        cont_ub=cont_ub,
        fitness_fn=fintess_fn,
        maximize=maximize
    )

    history = [global_best.reported_fit]
    n_evals = n_particles 
    sigma = (cont_ub - cont_lb) * 0.1 if n_continuous > 0 else np.array([]) # Standard deviation for Gaussian mutation

    for _ in range(1, max_iters + 1):
        new_pop = []
        for particle in pop:
            # Select a partner for crossover using tournament selection
            partner = tournament_selection(pop, k=3)
            
            # Continuous arithmetic -> Arithmetic crossover + Gaussian mutation
            if n_continuous > 0:
                alpha = np.random.rand(n_continuous)
                child_cont = alpha * particle.x_cont + (1 - alpha) * partner.x_cont
                if np.random.rand() < p_mut:
                    child_cont += np.random.randn(n_continuous) * sigma # Gaussian mutation
                child_cont = np.clip(child_cont, cont_lb, cont_ub) # Ensure continuous values stay within bounds
            else:
                child_cont = np.array([])
            
            # Discrete uniform -> Uniform crossover + Random reset mutation
            child_disc = particle.x_disc.copy()
            if np.random.rand() < p_cross:
                mask = np.random.rand(len(child_disc)) < 0.5
                child_disc[mask] = partner.x_disc[mask] # Uniform crossover
            for i in range(len(child_disc)):
                if np.random.rand() < p_mut:
                    child_disc[i] = np.random.randint(0, discrete_options[i]) # Random reset mutation

            # Create child particle and evaluate fitness
            child = HybridParticle(n_continuous=n_continuous, discrete_options=discrete_options, cont_lb=cont_lb, cont_ub=cont_ub)
            child.x_cont = child_cont
            child.x_disc = child_disc
            child.fitness = fintess_fn(child_cont, child_disc)
            n_evals += 1
            child.pb_cont = child_cont.copy()
            child.pb_disc = child_disc.copy()
            child.pb_fit = child.fitness
            
            # Greedy acceptance: keep the better particle (parent or child) based on fitness
            if maximize:
                new_pop.append(child if child.fitness > particle.fitness else particle)
            else:
                new_pop.append(child if child.fitness < particle.fitness else particle)
            
        pop = new_pop
        global_best.update_from_all(pop)
        history.append(global_best.reported_fit)

    return AlgorithmResult(
        name = "Pure GA",
        best_continuous=global_best.x_cont,
        best_discrete=global_best.x_disc,
        best_fitness=global_best.reported_fit,
        history=history,
        n_evaluations=n_evals,
        maximize=maximize
    )
