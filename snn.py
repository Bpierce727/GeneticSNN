# snn.py

import numpy as np
import logging
import uuid
import copy

# Configure logging for debugging and error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_id(prefix):
    """Generates a unique identifier with a given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

class Neuron:
    def __init__(self, neuron_id, neuron_type='excitatory', threshold=1.0, resting_potential=0.0,
                 reset_potential=0.0, tau=10.0, refractory_period=5.0):
        """
        Initialize a neuron with specified parameters.

        :param neuron_id: Unique identifier for the neuron
        :param neuron_type: Type of the neuron ('excitatory', 'inhibitory', 'input', 'hidden', 'output', or custom)
        :param threshold: Membrane potential threshold for spike generation
        :param resting_potential: Resting membrane potential
        :param reset_potential: Membrane potential after a spike
        :param tau: Membrane time constant
        :param refractory_period: Time period after a spike during which the neuron cannot spike again
        """
        self.id = neuron_id
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.resting_potential = resting_potential
        self.reset_potential = reset_potential
        self.tau = tau
        self.refractory_period = refractory_period
        self.V = resting_potential
        self.last_spike_time = -np.inf  # Initialize to negative infinity
        self.input_synapses = []
        self.spike_train = []

    def add_input_synapse(self, synapse):
        self.input_synapses.append(synapse)

    def reset(self):
        self.V = self.resting_potential
        self.spike_train = []
        self.last_spike_time = -np.inf

    def simulate(self, t, dt):
        # Check for refractory period
        if (t - self.last_spike_time) < self.refractory_period:
            return False  # Neuron is in refractory period

        # Sum input currents from all synapses
        I_syn = sum(synapse.get_current(t) for synapse in self.input_synapses)

        # Adjust input current based on neuron type
        if self.neuron_type == 'excitatory':
            I_syn = max(I_syn, 0)  # Excitatory neurons only receive positive inputs
        elif self.neuron_type == 'inhibitory':
            I_syn = min(I_syn, 0)  # Inhibitory neurons only receive negative inputs
        elif self.neuron_type == 'input':
            pass  # Input neurons handle inputs externally
        else:
            # Custom neuron types can define their own input handling
            pass

        # Update membrane potential using the leaky integrate-and-fire model
        dV = (-self.V + I_syn) * (dt / self.tau)
        self.V += dV

        # Check for spike
        if self.V >= self.threshold:
            # Spike occurs
            self.V = self.reset_potential
            self.spike_train.append(t)
            self.last_spike_time = t
            return True  # Spike occurred
        else:
            return False  # No spike

class Synapse:
    def __init__(self, synapse_id, pre_neuron_id, post_neuron_id, weight=1.0, delay=0.0, synapse_type='regular'):
        """
        Initialize a synapse with specified parameters.

        :param synapse_id: Unique identifier for the synapse
        :param pre_neuron_id: ID of the presynaptic neuron
        :param post_neuron_id: ID of the postsynaptic neuron
        :param weight: Synaptic weight
        :param delay: Transmission delay
        :param synapse_type: Type of the synapse ('regular', 'inhibitory', etc.)
        """
        self.id = synapse_id
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = weight
        self.delay = delay
        self.synapse_type = synapse_type
        self.pre_neuron = None  # To be linked after network construction

    def get_current(self, t):
        # Calculate synaptic current based on presynaptic spikes
        I = 0.0
        if self.pre_neuron is None:
            return I  # Synapse not linked properly
        for spike_time in self.pre_neuron.spike_train:
            arrival_time = spike_time + self.delay
            # Using a small tolerance for floating point comparison
            if np.isclose(arrival_time, t, atol=1e-3):
                I += self.weight
        return I

class Genome:
    def __init__(self, neurons=None, synapses=None):
        """
        Initialize the genome with neurons and synapses.

        :param neurons: List of neuron dictionaries
        :param synapses: List of synapse dictionaries
        """
        self.neurons = neurons if neurons is not None else []
        self.synapses = synapses if synapses is not None else []

    def mutate(self, mutation_rate=0.1):
        """
        Applies random mutations to the genome.

        :param mutation_rate: Probability of each gene being mutated
        """
        # Mutate neurons
        for neuron in self.neurons:
            if np.random.rand() < mutation_rate:
                # Example mutation: adjust threshold
                original_threshold = neuron['threshold']
                neuron['threshold'] += np.random.normal(0, 0.1)
                neuron['threshold'] = max(0.1, neuron['threshold'])  # Ensure threshold is positive
                logger.debug(f"Mutated neuron {neuron['id']} threshold from {original_threshold} to {neuron['threshold']}")

            if np.random.rand() < mutation_rate:
                # Example mutation: adjust tau
                original_tau = neuron['tau']
                neuron['tau'] += np.random.normal(0, 0.5)
                neuron['tau'] = max(0.1, neuron['tau'])  # Ensure tau is positive
                logger.debug(f"Mutated neuron {neuron['id']} tau from {original_tau} to {neuron['tau']}")

        # Mutate synapses
        for synapse in self.synapses:
            if np.random.rand() < mutation_rate:
                # Example mutation: adjust weight
                original_weight = synapse['weight']
                synapse['weight'] += np.random.normal(0, 0.1)
                synapse['weight'] = np.clip(synapse['weight'], -1.0, 1.0)  # Clamp weight
                logger.debug(f"Mutated synapse {synapse['id']} weight from {original_weight} to {synapse['weight']}")

            if np.random.rand() < mutation_rate:
                # Example mutation: adjust delay
                original_delay = synapse['delay']
                synapse['delay'] += np.random.normal(0, 0.1)
                synapse['delay'] = max(0.0, synapse['delay'])  # Ensure delay is non-negative
                logger.debug(f"Mutated synapse {synapse['id']} delay from {original_delay} to {synapse['delay']}")

    @staticmethod
    def crossover(parent1, parent2):
        """
        Performs meiosis-like crossover between two parent genomes to produce an offspring genome.

        :param parent1: Genome object
        :param parent2: Genome object
        :return: Offspring Genome object
        """
        offspring_neurons = []
        offspring_synapses = []

        # Create neuron mapping based on shared neuron IDs
        parent1_neuron_ids = set(neuron['id'] for neuron in parent1.neurons)
        parent2_neuron_ids = set(neuron['id'] for neuron in parent2.neurons)
        shared_neuron_ids = parent1_neuron_ids.intersection(parent2_neuron_ids)

        # Inherit shared neurons randomly from either parent
        for neuron_id in shared_neuron_ids:
            parent_choice = random.choice([parent1, parent2])
            neuron = next((n for n in parent_choice.neurons if n['id'] == neuron_id), None)
            if neuron:
                offspring_neurons.append(copy.deepcopy(neuron))

        # Inherit unique neurons from each parent
        unique_parent1_neurons = [n for n in parent1.neurons if n['id'] not in shared_neuron_ids]
        unique_parent2_neurons = [n for n in parent2.neurons if n['id'] not in shared_neuron_ids]

        # Randomly decide whether to include unique neurons
        for neuron in unique_parent1_neurons + unique_parent2_neurons:
            if random.random() < 0.5:
                offspring_neurons.append(copy.deepcopy(neuron))

        # Handle synapses
        # Create synapse mapping based on shared synapse IDs
        parent1_synapse_ids = set(syn['id'] for syn in parent1.synapses)
        parent2_synapse_ids = set(syn['id'] for syn in parent2.synapses)
        shared_synapse_ids = parent1_synapse_ids.intersection(parent2_synapse_ids)

        # Inherit shared synapses randomly from either parent
        for syn_id in shared_synapse_ids:
            parent_choice = random.choice([parent1, parent2])
            synapse = next((s for s in parent_choice.synapses if s['id'] == syn_id), None)
            if synapse:
                offspring_synapses.append(copy.deepcopy(synapse))

        # Inherit unique synapses from each parent
        unique_parent1_synapses = [s for s in parent1.synapses if s['id'] not in shared_synapse_ids]
        unique_parent2_synapses = [s for s in parent2.synapses if s['id'] not in shared_synapse_ids]

        for syn in unique_parent1_synapses + unique_parent2_synapses:
            if random.random() < 0.5:
                offspring_synapses.append(copy.deepcopy(syn))

        # Assign new unique IDs to any neurons or synapses that may have conflicting IDs
        # (This implementation assumes IDs are unique across parents)
        # Alternatively, implement ID translation or mapping

        offspring_genome = Genome(neurons=offspring_neurons, synapses=offspring_synapses)
        return offspring_genome

    def validate_and_repair_genome(neurons, synapses):
        """
        Validates the genome and repairs any inconsistencies, such as synapses referencing nonexistent neurons.

        :param neurons: List of neuron dictionaries
        :param synapses: List of synapse dictionaries
        :return: Tuple of (valid_neurons, valid_synapses)
        """
        valid_neuron_ids = set(neuron['id'] for neuron in neurons)
        valid_synapses = []
        for syn in synapses:
            if syn['pre_neuron_id'] in valid_neuron_ids and syn['post_neuron_id'] in valid_neuron_ids:
                valid_synapses.append(syn)
            else:
                logger.warning(f"Synapse {syn['id']} references invalid neurons and will be removed.")
        return neurons, valid_synapses

    def to_dict(genome):
        """Converts the genome to a dictionary representation."""
        return {
            'neurons': copy.deepcopy(genome.neurons),
            'synapses': copy.deepcopy(genome.synapses)
        }

    def from_dict(genome_dict):
        """Creates a Genome object from a dictionary representation."""
        neurons = copy.deepcopy(genome_dict.get('neurons', []))
        synapses = copy.deepcopy(genome_dict.get('synapses', []))
        return Genome(neurons=neurons, synapses=synapses)

class Network:
    def __init__(self):
        self.neurons = {}  # Key: neuron_id, Value: Neuron object
        self.synapses = {}  # Key: synapse_id, Value: Synapse object
        self.genome = Genome()

    def add_neuron(self, neuron):
        self.neurons[neuron.id] = neuron
        self.genome.neurons.append({
            'id': neuron.id,
            'type': neuron.neuron_type,
            'threshold': neuron.threshold,
            'resting_potential': neuron.resting_potential,
            'reset_potential': neuron.reset_potential,
            'tau': neuron.tau,
            'refractory_period': neuron.refractory_period
        })

    def add_synapse(self, synapse):
        self.synapses[synapse.id] = synapse
        self.genome.synapses.append({
            'id': synapse.id,
            'pre_neuron_id': synapse.pre_neuron_id,
            'post_neuron_id': synapse.post_neuron_id,
            'weight': synapse.weight,
            'delay': synapse.delay,
            'type': synapse.synapse_type
        })
        # Link synapse to presynaptic neuron
        if synapse.pre_neuron_id in self.neurons:
            synapse.pre_neuron = self.neurons[synapse.pre_neuron_id]
        else:
            logger.error(f"Pre neuron {synapse.pre_neuron_id} not found for synapse {synapse.id}")

    def reset(self):
        for neuron in self.neurons.values():
            neuron.reset()

    def simulate(self, t_max=100.0, dt=1.0):
        times = np.arange(0, t_max, dt)
        for t in times:
            for neuron in self.neurons.values():
                neuron.simulate(t, dt)

    def get_genome(self):
        """
        Extracts the network's genome representation.
        """
        return to_dict(self.genome)

    @classmethod
    def from_genome(cls, genome):
        """
        Recreates a network from a genome representation.
        Includes robust error handling to manage invalid genome data.

        :param genome: Dictionary containing 'neurons' and 'synapses' lists
        :return: Network object
        """
        network = cls()
        genome_obj = from_dict(genome)
        genome_obj.neurons, genome_obj.synapses = validate_and_repair_genome(genome_obj.neurons, genome_obj.synapses)

        # Create neurons
        for neuron_params in genome_obj.neurons:
            try:
                neuron = Neuron(
                    neuron_id=neuron_params['id'],
                    neuron_type=neuron_params.get('type', 'excitatory'),
                    threshold=float(neuron_params.get('threshold', 1.0)),
                    resting_potential=float(neuron_params.get('resting_potential', 0.0)),
                    reset_potential=float(neuron_params.get('reset_potential', 0.0)),
                    tau=float(neuron_params.get('tau', 10.0)),
                    refractory_period=float(neuron_params.get('refractory_period', 5.0))
                )
                network.neurons[neuron.id] = neuron
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error processing neuron {neuron_params.get('id', 'Unknown')}: {e}")
                continue

        # Create synapses
        for synapse_params in genome_obj.synapses:
            try:
                pre_id = synapse_params['pre_neuron_id']
                post_id = synapse_params['post_neuron_id']
                synapse = Synapse(
                    synapse_id=synapse_params['id'],
                    pre_neuron_id=pre_id,
                    post_neuron_id=post_id,
                    weight=float(synapse_params.get('weight', 1.0)),
                    delay=float(synapse_params.get('delay', 0.0)),
                    synapse_type=synapse_params.get('type', 'regular')
                )
                network.synapses[synapse.id] = synapse
                # Link synapse to presynaptic neuron
                if pre_id in network.neurons:
                    synapse.pre_neuron = network.neurons[pre_id]
                else:
                    logger.error(f"Pre neuron {pre_id} not found for synapse {synapse.id}")
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error processing synapse {synapse_params.get('id', 'Unknown')}: {e}")
                continue

        network.genome = genome_obj  # Assign the genome
        return network
