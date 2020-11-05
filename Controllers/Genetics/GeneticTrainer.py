import torch
import random
from typing import List
import concurrent.futures
from Controllers.Genetics.SnakeWorker import SnakeWorker
from Views.ProgressBar import ProgressBar
from Models.SnakeBoard import SnakeBoard
from Models.Snake import Snake
from Models.Color import Color


class GeneticTrainer:

    @staticmethod
    def train(get_model, initial_state_dict, population: int = 1024,
              generations: int = 256, workers: int = 8, mutation_rate=0.05) -> (List[float], dict):
        """Trains the given `model` for a given number of `generations` with asynchronous `workers`.

        Args:
            get_model: The function to get a PyTorch model to train. Must also conform to SnakeControllable.
            initial_state_dict: The initial state dictionary to use with the model.
            population (int): The number of snakes total.
            generations (int): The total number of crossover events to have.
            workers (int): The number of concurrent threads with which to train.
            mutation_rate (float): The odds of randomly mutating a state_dict value.
        """
        fitness_history = []
        progress_bar = ProgressBar()
        best_fitness: float = 0.0
        best_state_dict: dict = initial_state_dict.copy()
        for generation in range(generations):
            # A list of tuples: (fitness: float, state_dict: dict)
            training_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Append all of the training results and wait for completion
                for i in range(workers):
                    model = get_model()
                    model.load_state_dict(best_state_dict.copy())
                    executor.submit(lambda: training_results.append(
                        SnakeWorker.simulate(model, population, mutation_rate=mutation_rate)))
            # Take the highest two and crossover
            executor.shutdown(wait=True)
            training_results.sort(key=lambda result: result[0], reverse=True)
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
        return fitness_history, best_state_dict

    @staticmethod
    def startSimulation(get_model, initial_state_dict, population: int = 512,
                        generations: int = 128, mutation_rate: float = 0.05) -> (dict, List[float]):
        """Starts the simulations using a large population-child setup.

        Args:
            get_model: A function that returns a PyTorch model to use. Must conform to SnakeControllable.
            initial_state_dict: The initial state dictionary to use.
            population (int): The population size to use for each generation.
            generations (int): The total number of times to repeat the child process
            mutation_rate (float): How often to mutate individual numbers in state.

        Returns:
            (dict, List[float]): The best state dictionary and the fitness history over training.
        """
        states = [GeneticTrainer.randomize(initial_state_dict, mutation_rate) for _ in range(population)]
        best_dict = {}
        fitness_history = []
        for generation in range(generations):
            best_dict, best_fitness, next_population = GeneticTrainer.simulate(get_model(), states,
                                                                               population, mutation_rate)
            fitness_history.append(best_fitness)
            states = next_population
        return best_dict, fitness_history

    @staticmethod
    def simulate(model, state_dicts, population, mutation_rate) -> (dict, float, List[dict]):
        """Simulates the snake population with the given `state_dicts`.

        Returns:
            (dict, float, List[dict]): The best snake, its fitness, and the next state dictionary populations to use.
        """
        new_state_dicts = state_dicts.copy()
        for state in new_state_dicts:
            GeneticTrainer.randomize(state, mutation_rate)
        # Run the simulations
        best_fitness: float = 0.0
        avg_fitness = 0.0
        fitness_history = []
        best_state_dict = state_dicts[0]
        for i in range(population):
            score, move_counter = GeneticTrainer.playGame(model, new_state_dicts[i])
            fitness = GeneticTrainer.fitness(score, move_counter)
            avg_fitness += fitness
            fitness_history.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_state_dict = new_state_dicts[i]

        # Create children from those snakes
        avg_fitness /= population
        next_population = []

        # Add the snakes that were above average fitness
        for i in range(population):
            if fitness_history[i] > avg_fitness:
                next_population.append(state_dicts[i])

        # Append any children
        parent_count = len(next_population)
        for i in range(population - len(next_population)):
            parent1 = next_population[random.randint(0, parent_count - 1)]
            parent2 = next_population[random.randint(0, parent_count - 1)]
            child = GeneticTrainer.crossover(parent1, parent2, mutation_rate)
            next_population.append(child)
        return best_state_dict, best_fitness, next_population

    @staticmethod
    def playGame(model, state_dict) -> (int, int):
        """Plays a game with the given board and returns the score and total moves."""
        move_counter = 0
        score = 0
        model.load_state_dict(state_dict)
        snakeBoard = GeneticTrainer.generateDefaultBoard(controller=model)
        snake = snakeBoard.snakeDict.get('AI')
        while not snakeBoard.isGameOver():
            score = snake.score
            snakeBoard.update()
            move_counter += 1
        return score, move_counter

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

    @staticmethod
    def randomize(state_dict: dict, mutation_rate: float):
        """Randomizes the `state_dict` randomly based on `percent_variation`."""
        for key in state_dict.keys():
            with torch.no_grad():
                # Flatten mask and then reshape it
                matrix = torch.flatten(state_dict[key])
                mask = torch.flatten(torch.ones(matrix.size()))
                for i in range(len(matrix)):
                    if random.random() < mutation_rate:
                        matrix[i] = 2 * random.random() - 1
                matrix = torch.reshape(matrix, state_dict[key].size())
                state_dict[key] = matrix

    @staticmethod
    def fitness(score: int, turns: int) -> float:
        """Returns the snake fitness based on the moves given."""
        return 100 * score + turns

    @staticmethod
    def generateDefaultBoard(controller) -> SnakeBoard:
        """Generates an example `SnakeBoard` to use for training."""
        snakeBoard = SnakeBoard()
        snake = Snake(name='AI', mark=Color.colorize('X', Color.cyan),
                      segments=[(snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 3),
                                (snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 2),
                                (snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 1)],
                      controller=controller, maxHealth=50)
        snakeBoard.addSnake(snake)
        snakeBoard.generatePoint()
        return snakeBoard


def main():
    # Ensure that crossover works properly.
    # There should be a mix of 1's, 2's, and potentially a random number or more.
    dict1 = {'weights': torch.Tensor([[1, 1, 1], [1, 1, 1]])}
    dict2 = {'weights': torch.Tensor([[2, 2, 2], [2, 2, 2]])}
    crossover = GeneticTrainer.crossover(dict1, dict2, mutation_rate=0.4)
    print(crossover.get('weights'))


if __name__ == '__main__':
    main()
