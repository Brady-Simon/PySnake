import torch
import random
from threading import Thread
import concurrent.futures
from Controllers.Genetics.SnakeWorker import SnakeWorker


class GeneticTrainer:

    @staticmethod
    def train(model, steps: int = 1024, generations: int = 256, workers: int = 8, mutation_rate=0.05):
        """Trains the given `model` for a given number of `generations` with asynchronous `workers`."""
        training_results = []
        for generation in range(generations):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Append all of the training results and wait for completion
                executor.map(lambda: training_results.append(
                    SnakeWorker.simulate(model, steps, mutation_rate=mutation_rate)), range(workers))
                executor.shutdown(wait=True)
                # Take the highest two and crossover
                # TODO: Pick highest and crossover

    @staticmethod
    def crossover(state_dict1, state_dict2, mutation_rate: float = 0.05) -> dict:
        """Returns the hybrid of the two state dictionaries.
        Ensure that the two state dictionaries and values are the same size.
        """
        crossover_result = state_dict1.copy()
        parent = state_dict2.copy()
        for key in state_dict1.keys():
            with torch.no_grad():
                # Flatten mask and then reshape it
                matrix = torch.flatten(crossover_result[key].copy())
                """The matrix to modify, based of `state_dict1`"""
                matrix2 = torch.flatten(parent[key])
                """The matrix to mix genes with `matrix`."""
                for i in range(len(matrix)):
                    if random.random() < mutation_rate:
                        # Randomly mutate value between -1 and 1
                        matrix[i] = 2 * random.random() - 1
                    elif random.random() < 0.5:
                        # Replace value with gene from parent
                        matrix[i] = matrix2[i]

                # Reshape the matrix to its original shape
                matrix = matrix.reshape(crossover_result[key].size())
                crossover_result[key] = matrix

        # Return the resulting state dictionary
        return crossover_result
