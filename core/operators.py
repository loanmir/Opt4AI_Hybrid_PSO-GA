"""
    Update operators:
         - pso_update: updates the velocity and position of a particle in the continuous space using the PSO formula.
         - ga_update: evolves the discrete dimensions using crossover + mutation
"""

import numpy as np
from core.particle import HybridParticle





def pso_update(
       particle: HybridParticle, # Particle to update
       global_best_cont: np.ndarray, # Global best position in the continuous space
       w: float = 0.7, # Inertia weight
       c1: float = 1.5, # Cognitive coefficient
       c2: float = 1.5,  # Social coefficient
       v_clamp_factor: float = 0.2 # Clamping factor for velocity to prevent overshooting
) -> None:
    """ 
        Standard PSO velocity + position update on the continuous part only 
        Velocity Clamping: To prevent particles from moving too fast and potentially overshooting good solutions, we can clamp the velocity to a certain range.
        
        v ← w·v + c1·r1·(pb - x) + c2·r2·(gb - x)
        x ← x + v
    """
    n = len(particle.x_cont)
    if n == 0:
        return  # No continuous dimensions to update - Pure discrete problem
    r1 = np.random.random(n)
    r2 = np.random.random(n)

    particle.v_cont = w * particle.v_cont + c1 * r1 * (particle.pb_cont - particle.x_cont) + c2 * r2 * (global_best_cont - particle.x_cont)
    
    # Velocity Clamping
    # In landscapes with steep penalties, PSO particles accelerate rapidly towards the global best, which can lead to overshooting and divergence. 
    # By clamping the velocity, we can prevent particles from moving too fast and potentially overshooting good solutions and go past them.
    # Actually imposing a strict speed limit in order for the particles to take smaller and controlled steps.
    lb, ub = particle.cont_lb, particle.cont_ub
    v_max = (ub - lb) * v_clamp_factor
    particle.v_cont = np.clip(particle.v_cont, -v_max, v_max)

    # Update position
    particle.x_cont = particle.x_cont + particle.v_cont

    
    # Producing boolean arrays to identify which particles have violated the bounds
    hit_lower_bound = particle.x_cont < lb
    hit_upper_bound = particle.x_cont > ub
    particle.x_cont = np.clip(particle.x_cont, lb, ub) # Forcing every element of x_cont to be within the bounds [lb, ub]
    particle.v_cont[hit_lower_bound | hit_upper_bound] *= 0.5 # Combining two boolean arrays with OR to identify which particles have violated either the lower or upper bound, 
    # and halving the velocity of those particles to reduce the chance of future violations












def ga_update(
        particle: HybridParticle, # Particle to update
        partner: HybridParticle, # Partner particle for crossover
        fitness_fn, # Fitness function to evaluate the new candidate solution
        p_crossover: float = 0.7, # Crossover probability
        p_mutation: float = 0.1, # Mutation probability
        maximize: bool = False # Whether we are maximizing or minimizing the fitness
) -> None:
    """
        Uniform crossover + mutation for the discrete dimensions only. 
        Crossover is performed with a randomly selected partner particle
        Mutation is applied with a certain probability to introduce diversity.

        GREEDYNESS: The new candidate solution is accepted if it has better fitness than the current particle (for maximization) or worse fitness (for minimization).
    """

    n = len(particle.x_disc)
    if n == 0:
        return  # No discrete dimensions to update - Pure continuous problem
    child = particle.x_disc.copy()

    # Uniform crossover
    if np.random.rand() < p_crossover:
        mask = np.random.rand(n) < 0.5 #50/50 -> UNIFORM CROSSOVER
        child[mask] = partner.x_disc[mask]

    # Mutation
    for i in range(n):
        if np.random.rand() < p_mutation:
            # Randomly select a new value for this discrete dimension
            child[i] = np.random.randint(0, particle.discrete_options[i]) # Noise injection
    
    # If nothing actually changed, there's nothing to evaluate or compare
    if np.array_equal(child, particle.x_disc):
        return

    # Greedy acceptance check 
    candidate_fit = fitness_fn(particle.x_cont, child)
    is_better_or_equal = (
        candidate_fit >= particle.fitness if maximize
        else candidate_fit <= particle.fitness
    )

    if is_better_or_equal:
        particle.x_disc = child
        particle.fitness = candidate_fit
    # else: discard the candidate, keep particle.x_disc unchanged
    
    #particle.x_disc = child










def tournament_selection(
        population: list[HybridParticle], # List of particles in the current population
        k: int = 3 # Number of particles to compete in each tournament
) -> HybridParticle:
    """
        Tournament selection: Taking k random particles, return the fittest
        Used to pick GA crossover partner
    """

    candidates = np.random.choice(population, size=k, replace=False)
    return max(candidates, key=lambda p: p.fitness)