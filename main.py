# main.py

import pygame
import numpy as np
import random
import math
import uuid
import logging

from snn import Neuron, Synapse, Network, generate_unique_id, from_dict, to_dict
from genetic import GeneticAlgorithm
from game import GameEnvironment, Agent, Food, line_circle_intersection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_initial_network():
    """
    Creates an initial neural network for the agent with the specified input and output neurons.
    This function is used by the GeneticAlgorithm to initialize new networks.
    """
    network = Network()

    # Create input neurons (5)
    input_neurons = []
    for _ in range(5):
        neuron_id = generate_unique_id('N')
        neuron = Neuron(
            neuron_id=neuron_id,
            neuron_type='input',
            threshold=1.0,
            tau=10.0,
            refractory_period=5.0
        )
        network.add_neuron(neuron)
        input_neurons.append(neuron)

    # Create output neurons (3)
    output_neurons = []
    for _ in range(3):
        neuron_id = generate_unique_id('N')
        neuron = Neuron(
            neuron_id=neuron_id,
            neuron_type='output',
            threshold=1.0,
            tau=10.0,
            refractory_period=5.0
        )
        network.add_neuron(neuron)
        output_neurons.append(neuron)

    # Create synapses between input and output neurons randomly
    all_neurons = input_neurons + output_neurons
    for pre_neuron in all_neurons:
        for post_neuron in output_neurons:
            if pre_neuron != post_neuron and random.random() < 0.5:
                synapse_id = generate_unique_id('S')
                synapse = Synapse(
                    synapse_id=synapse_id,
                    pre_neuron_id=pre_neuron.id,
                    post_neuron_id=post_neuron.id,
                    weight=random.uniform(-1.0, 1.0),
                    delay=random.uniform(0.0, 5.0),
                    synapse_type='regular'
                )
                network.add_synapse(synapse)

    return network

def main():
    pygame.init()

    # Screen dimensions
    width = 800
    height = 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SNN Agent Simulation")

    # Create the game environment
    game_env = GameEnvironment(width, height)

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(population_size=20, mutation_rate=0.1, elite_size=2)
    ga.initialize_population(create_initial_network)

    # Spawn agents based on initial population
    for network in ga.population:
        agent_id = generate_unique_id('Agent')
        position = [random.uniform(0, width), random.uniform(0, height)]
        angle = random.uniform(0, 360)
        agent = Agent(position, angle, network, agent_id)
        game_env.agents.append(agent)

    # Spawn initial food
    for _ in range(20):
        game_env.spawn_food()

    clock = pygame.time.Clock()

    generation = 1
    running = True
    while running:
        dt = clock.tick(30) / 1000.0  # Convert milliseconds to seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update the game environment
        game_env.update(dt)

        # Check if all agents are dead to proceed to the next generation
        if not game_env.agents:
            logger.info(f"Generation {generation} complete. All agents have died.")
            # Evaluate fitness and create next generation
            ga.create_next_generation(game_env, create_initial_network)
            # Clear the environment
            game_env.agents = []
            game_env.foods = []
            # Spawn new agents
            for network in ga.population:
                agent_id = generate_unique_id('Agent')
                position = [random.uniform(0, width), random.uniform(0, height)]
                angle = random.uniform(0, 360)
                agent = Agent(position, angle, network, agent_id)
                game_env.agents.append(agent)
            # Spawn new food
            for _ in range(20):
                game_env.spawn_food()
            # Increment generation counter
            generation += 1
            logger.info(f"Starting Generation {generation}.")

        # Clear the screen
        screen.fill((255, 255, 255))

        # Draw food
        for food in game_env.foods:
            pygame.draw.circle(screen, (0, 255, 0), food.position.astype(int), food.radius)

        # Draw agents
        for agent in game_env.agents:
            # Agent color based on hunger level
            if agent.hunger_level == 0:
                color = (0, 0, 255)  # Blue
            elif agent.hunger_level == 1:
                color = (0, 100, 255)  # Light Blue
            elif agent.hunger_level == 2:
                color = (0, 150, 200)  # Teal
            else:
                color = (0, 200, 150)  # Darker Teal

            pygame.draw.circle(screen, color, agent.position.astype(int), agent.radius)

            # Draw agent's direction
            rad_angle = math.radians(agent.angle)
            end_pos = agent.position + np.array([math.cos(rad_angle), -math.sin(rad_angle)]) * agent.radius
            pygame.draw.line(screen, (0, 0, 0), agent.position.astype(int), end_pos.astype(int), 2)

        # Update the display
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
