# game.py

import pygame
import numpy as np
import random
import math
import uuid
import logging

from snn import Neuron, Synapse, Network, generate_unique_id, Genome
from genetic import GeneticAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Food:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.radius = 5  # Radius of the food circle

class Agent:
    def __init__(self, position, angle, network, agent_id):
        self.position = np.array(position, dtype=float)
        self.angle = angle  # In degrees
        self.network = network
        self.hunger_level = 0  # Hunger level ranges from 0 to 3
        self.agent_id = agent_id
        self.radius = 10  # Radius of the agent circle
        self.speed = 2.0  # Movement speed
        self.rotation_speed = 5.0  # Rotation speed in degrees per update
        self.alive = True  # Agent is alive
        self.last_hunger_increase_time = 0  # Time when hunger was last increased

    def sense(self, game_env):
        """
        Updates the agent's neural network inputs based on the environment.
        """
        # Reset network inputs
        for neuron in self.network.neurons.values():
            neuron.reset()

        # Input neurons (assumed to be the first 5 neurons)
        input_neurons = [neuron for neuron in self.network.neurons.values() if neuron.neuron_type == 'input']
        if len(input_neurons) < 5:
            logger.warning(f"Agent {self.agent_id} has insufficient input neurons.")
            return

        # Sensor 1: Food detected in front
        if self.detect_food(game_env):
            input_neurons[0].V = input_neurons[0].threshold  # Excite the neuron

        # Sensor 2: Wall detected in front
        if self.detect_wall(game_env):
            input_neurons[1].V = input_neurons[1].threshold  # Excite the neuron

        # Sensor 3: Nothing detected in front
        if not self.detect_food(game_env) and not self.detect_wall(game_env):
            input_neurons[2].V = input_neurons[2].threshold  # Excite the neuron

        # Sensor 4: Hunger level over 1
        if self.hunger_level > 1:
            input_neurons[3].V = input_neurons[3].threshold  # Excite the neuron

        # Sensor 5: Hunger level over 2
        if self.hunger_level > 2:
            input_neurons[4].V = input_neurons[4].threshold  # Excite the neuron

    def update(self, dt, game_env):
        """
        Updates the agent's state based on neural network outputs.
        """
        # Update neural network
        self.network.simulate(t_max=dt, dt=dt)

        # Output neurons (assumed to be the last 3 neurons)
        output_neurons = [neuron for neuron in self.network.neurons.values() if neuron.neuron_type == 'output']
        if len(output_neurons) < 3:
            logger.warning(f"Agent {self.agent_id} has insufficient output neurons.")
            return

        # Check output neurons for spikes
        move_forward = any(spike_time >= 0 for spike_time in output_neurons[0].spike_train)
        turn_clockwise = any(spike_time >= 0 for spike_time in output_neurons[1].spike_train)
        turn_anticlockwise = any(spike_time >= 0 for spike_time in output_neurons[2].spike_train)

        # Apply movements based on outputs
        if move_forward:
            self.move_forward()

        if turn_clockwise:
            self.angle += self.rotation_speed * dt

        if turn_anticlockwise:
            self.angle -= self.rotation_speed * dt

        # Ensure angle stays within [0, 360)
        self.angle %= 360

        # Check for collisions with food
        self.check_food_collision(game_env)

    def move_forward(self):
        rad_angle = math.radians(self.angle)
        direction = np.array([math.cos(rad_angle), -math.sin(rad_angle)])
        self.position += direction * self.speed

    def detect_food(self, game_env):
        """
        Detects if there is food in front of the agent within a certain range.
        """
        rad_angle = math.radians(self.angle)
        ray_direction = np.array([math.cos(rad_angle), -math.sin(rad_angle)])
        ray_start = self.position
        ray_end = ray_start + ray_direction * 50  # Raycast length

        for food in game_env.foods:
            if line_circle_intersection(ray_start, ray_end, food.position, food.radius):
                return True
        return False

    def detect_wall(self, game_env):
        """
        Detects if there is a wall in front of the agent within a certain range.
        """
        rad_angle = math.radians(self.angle)
        ray_direction = np.array([math.cos(rad_angle), -math.sin(rad_angle)])
        ray_start = self.position
        ray_end = ray_start + ray_direction * 50  # Raycast length

        # Check if ray_end is outside the map boundaries
        if not (0 <= ray_end[0] <= game_env.width) or not (0 <= ray_end[1] <= game_env.height):
            return True
        return False

    def check_food_collision(self, game_env):
        """
        Checks if the agent is colliding with any food.
        """
        for food in game_env.foods:
            distance = np.linalg.norm(self.position - food.position)
            if distance < self.radius + food.radius:
                # Consume the food
                game_env.foods.remove(food)
                self.hunger_level = max(self.hunger_level - 1, 0)
                break  # Only consume one food at a time

    def increase_hunger(self):
        """
        Increases the agent's hunger level and checks for starvation.
        """
        self.hunger_level += 1
        if self.hunger_level > 3:
            # Agent starves and dies
            self.alive = False

    def stay_within_bounds(self, game_env):
        """
        Keeps the agent within the map boundaries.
        """
        self.position[0] = np.clip(self.position[0], 0, game_env.width)
        self.position[1] = np.clip(self.position[1], 0, game_env.height)

def line_circle_intersection(p1, p2, center, radius):
    """
    Checks if a line segment (p1 to p2) intersects with a circle (center, radius).
    """
    d = p2 - p1
    f = p1 - center

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
    return False

class GameEnvironment:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.agents = []
        self.foods = []
        self.time = 0  # Simulation time

    def spawn_agent(self, create_initial_network_func):
        """
        Spawns a new agent with a unique network.

        :param create_initial_network_func: Function to create an initial network
        """
        position = [random.uniform(0, self.width), random.uniform(0, self.height)]
        angle = random.uniform(0, 360)
        network = create_initial_network_func()
        agent_id = generate_unique_id('Agent')
        agent = Agent(position, angle, network, agent_id)
        self.agents.append(agent)
        logger.info(f"Spawned agent {agent_id} at position {position} with angle {angle}.")

    def spawn_food(self, position=None):
        """
        Spawns a new food item at the specified position or randomly if not provided.

        :param position: Tuple or list of (x, y) coordinates
        """
        if position is None:
            position = [random.uniform(0, self.width), random.uniform(0, self.height)]
        food = Food(position)
        self.foods.append(food)
        logger.info(f"Spawned food at position {position}.")

    def update(self, dt):
        self.time += dt

        # Increase hunger every 30 seconds
        if int(self.time) % 30 == 0:
            for agent in self.agents:
                if int(self.time) != agent.last_hunger_increase_time:
                    agent.increase_hunger()
                    agent.last_hunger_increase_time = int(self.time)
                    if not agent.alive:
                        logger.info(f"Agent {agent.agent_id} has starved and died.")

        # Update agents
        for agent in self.agents[:]:  # Copy the list to avoid issues when removing agents
            if agent.alive:
                agent.sense(self)
                agent.update(dt, self)
                agent.stay_within_bounds(self)
            else:
                # Agent has died
                self.agents.remove(agent)
                logger.info(f"Removed agent {agent.agent_id} from the environment.")

        # Spawn food periodically (every 15 seconds) and limit total food
        if int(self.time) % 15 == 0 and len(self.foods) < 20:
            self.spawn_food()

    def create_initial_network(self):
        """
        Creates an initial neural network for the agent with the specified input and output neurons.
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
