import torch
import random
from typing import List
import concurrent.futures
from Controllers.Genetics.SnakeWorker import SnakeWorker
from Views.ProgressBar import ProgressBar


class GeneticTrainer:

    @staticmethod
    def train(model, steps: int = 1024, generations: int = 256, workers: int = 8, mutation_rate=0.05) -> List[float]:
        """Trains the given `model` for a given number of `generations` with asynchronous `workers`.

        Args:
            model: The PyTorch model to train. Must also conform to SnakeControllable.
            steps (int): The number of times to repeat individual mutation.
            generations (int): The total number of crossover events to have.
            workers (int): The number of concurrent threads with which to train.
            mutation_rate (float): The odds of randomly mutating a state_dict value.
        """
        fitness_history = []
        progress_bar = ProgressBar()
        best_fitness: float = 0.0
        best_state_dict: dict = {}
        for generation in range(generations):
            # A list of tuples: (fitness: float, state_dict: dict)
            training_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Append all of the training results and wait for completion
                for i in range(workers):
                    executor.submit(lambda: training_results.append(
                        SnakeWorker.simulate(model, steps, mutation_rate=mutation_rate)))
                # executor.map(lambda: training_results.append(
                #     SnakeWorker.simulate(model, steps, mutation_rate=mutation_rate)), range(workers))
            # Take the highest two and crossover
            executor.shutdown(wait=True)
            training_results.sort(key=lambda result: result[0])
            first_place = training_results[0]
            second_place = training_results[1]
            crossover_dict = GeneticTrainer.crossover(first_place[1], second_place[1], mutation_rate)
            # Update the model's state dictionary and continue the cycle
            if first_place[0] > best_fitness:
                # The current model outperformed the best model
                best_fitness = first_place[0]
                best_state_dict = first_place[1]
                model.load_state_dict(best_state_dict)
            else:
                # Continue to use the crossover model until performance is beaten
                model.load_state_dict(crossover_dict)
            fitness_history.append(first_place[0])
            progress_bar.printProgress(generation + 1, generations, 16)
        # Return the fitness history for graphing
        return fitness_history

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
                matrix = torch.flatten(crossover_result[key])
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
