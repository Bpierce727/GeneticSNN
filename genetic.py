# genetic.py

import numpy as np
import random
import copy
import logging

from snn import Genome, Network, generate_unique_id, from_dict, to_dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(self, population_size=20, mutation_rate=0.1, elite_size=2):
        """
        Initialize the genetic algorithm parameters.

        :param population_size: Number of individuals in the population
        :param mutation_rate: Probability of each gene being mutated
        :param elite_size: Number of top individuals to carry over to the next generation
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = []  # List of Network objects

    def initialize_population(self, create_initial_network_func):
        """
        Initialize the population with randomly generated networks.

        :param create_initial_network_func: Function to create an initial network
        """
        self.population = []
        for _ in range(self.population_size):
            network = create_initial_network_func()
            self.population.append(network)
        logger.info(f"Initialized population with {len(self.population)} individuals.")

    def evaluate_fitness(self, network, game_env):
        """
        Evaluate the fitness of a network based on its performance in the game environment.

        :param network: Network object
        :param game_env: GameEnvironment object
        :return: Fitness score (higher is better)
        """
        # Placeholder fitness function:
        # Fitness is based on the agent's hunger level and survival time
        # Modify this based on your specific game mechanics
        fitness = 0
        for agent in game_env.agents:
            if agent.network == network:
                fitness += (100 - agent.hunger_level) + len(agent.spike_train)
        return fitness

    def select_parents(self, fitness_scores):
        """
        Select parents based on fitness scores using roulette wheel selection.

        :param fitness_scores: List of fitness scores
        :return: Two parent Network objects
        """
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # Avoid division by zero
            return random.sample(self.population, 2)

        selection_probs = [f / total_fitness for f in fitness_scores]
        parents = np.random.choice(self.population, size=2, replace=False, p=selection_probs)
        return parents

    def create_next_generation(self, game_env, create_initial_network_func):
        """
        Create the next generation through selection, crossover, and mutation.

        :param game_env: GameEnvironment object for fitness evaluation
        :param create_initial_network_func: Function to create an initial network
        """
        # Evaluate fitness for each network
        fitness_scores = [self.evaluate_fitness(network, game_env) for network in self.population]

        # Select elite individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        elites = [self.population[i] for i in elite_indices]

        # Create new population
        new_population = elites.copy()

        # Generate offspring until reaching population size
        while len(new_population) < self.population_size:
            # Select parents
            parent1, parent2 = self.select_parents(fitness_scores)

            # Crossover
            offspring_genome = Genome.crossover(parent1.genome, parent2.genome)

            # Mutate
            offspring_genome.mutate(self.mutation_rate)

            # Validate and repair
            offspring_genome.neurons, offspring_genome.synapses = validate_and_repair_genome(
                offspring_genome.neurons, offspring_genome.synapses
            )

            # Create offspring network
            offspring_network = Network.from_genome(to_dict(offspring_genome))

            # Add to new population
            new_population.append(offspring_network)

        self.population = new_population[:self.population_size]
        logger.info("Created next generation.")

    def get_best_network(self, game_env):
        """
        Retrieve the best network based on fitness.

        :param game_env: GameEnvironment object
        :return: Network object with the highest fitness
        """
        fitness_scores = [self.evaluate_fitness(network, game_env) for network in self.population]
        best_index = np.argmax(fitness_scores)
        return self.population[best_index], fitness_scores[best_index]
