import copy
import torch
import random
import time
from typing import List, Tuple
from Views.ProgressBar import ProgressBar
from Models.SnakeBoard import SnakeBoard
from Models.Snake import Snake
from Models.Color import Color
from Models.Direction import Direction


class GeneticTrainer:

    @staticmethod
    def startSimulation(get_model, initial_state_dict, population: int = 512,
                        generations: int = 128, mutation_rate: float = 0.05,
                        cutoff_fitness: float = None, timeout: float = None) -> Tuple[dict, List[float]]:
        """Starts the simulations using a large population-child setup.

        Args:
            get_model: A function that returns a PyTorch model to use. Must conform to SnakeControllable.
            initial_state_dict: The initial state dictionary to use.
            population (int): The population size to use for each generation.
            generations (int): The total number of times to repeat the child process
            mutation_rate (float): How often to mutate individual numbers in state.
            cutoff_fitness (float): Stops training once the desired fitness is reached. Optional.
            timeout (float): Stops training after total time exceeds this value. Optional.

        Returns:
            (dict, List[float]): The best state dictionary and the fitness history over training.
        """
        progress_bar = ProgressBar()
        print('\r' + progress_bar.getProgressBar(0, generations), end='')
        # Mutate the initial state dictionary to get a diverse population.
        states = [copy.deepcopy(initial_state_dict) for _ in range(population)]
        for state in states:
            GeneticTrainer.randomize(state, mutation_rate)
        # Ensure the initial snake is kept intact.
        states[0] = copy.deepcopy(initial_state_dict)

        # Some important properties to keep track of.
        max_dict = {}
        max_fitness = 0.0
        fitness_history = []
        start_time = time.time()

        # Keep iterating through generations of snakes
        for generation in range(generations):
            best_dict, best_fitness, next_population = GeneticTrainer.simulate(get_model(), states,
                                                                               population, mutation_rate)
            fitness_history.append(best_fitness)
            total_time = time.time() - start_time
            if cutoff_fitness is not None and best_fitness >= cutoff_fitness:
                # We're done here, return
                print(f"\nCut-off fitness reached ({best_fitness} of {cutoff_fitness})")
                return best_dict, fitness_history
            elif best_fitness > max_fitness:
                # New record!
                max_dict = best_dict
                max_fitness = best_fitness

            # Return if the training time exceeds timeout
            if timeout is not None and total_time >= timeout:
                print(f"\n{timeout} second timeout reached.")
                return max_dict, fitness_history

            # Update the current population of snakes and print current progress
            states = next_population
            print('\r' + progress_bar.getProgressBar(generation + 1, generations) + f' Fitness: {best_fitness}', end='')

        print()
        return max_dict, fitness_history

    @staticmethod
    def simulate(model, state_dicts, population, mutation_rate) -> Tuple[dict, float, List[dict]]:
        """Simulates the snake population with the given `state_dicts`.

        Returns:
            (dict, float, List[dict]): The best snake, its fitness, and the next state dictionary populations to use.
        """
        # Copy over the generation
        new_state_dicts = [copy.deepcopy(state) for state in state_dicts]
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
                next_population.append(copy.deepcopy(state_dicts[i]))

        # Ensure that there are enough parents by adding randomized parents if necessary.
        while len(next_population) < population//2:
            mutated_parent = copy.deepcopy(state_dicts[random.randint(0, len(state_dicts) - 1)])
            GeneticTrainer.randomize(mutated_parent, mutation_rate=mutation_rate)
            next_population.append(mutated_parent)

        # Append any children
        parent_count = len(next_population)
        for i in range(population - len(next_population)):
            parent1 = next_population[random.randint(0, parent_count - 1)]
            parent2 = next_population[random.randint(0, parent_count - 1)]
            child = GeneticTrainer.crossover(parent1, parent2, mutation_rate)
            next_population.append(child)
        return best_state_dict, best_fitness, next_population

    @staticmethod
    def playGame(model, state_dict) -> Tuple[int, int]:
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

        crossover_result = copy.deepcopy(state_dict1)
        parent = copy.deepcopy(state_dict2)
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
    def generateDefaultBoard(controller, adaptiveHealth: bool = False) -> SnakeBoard:
        """Generates an example `SnakeBoard` to use for training."""
        snakeBoard = SnakeBoard()
        segments = GeneticTrainer.generateSnakeSegments(snakeBoard.board,
                                                        snakeBoard.board.columns(),
                                                        snakeBoard.board.rows())

        snake = Snake(name='AI', mark=Color.colorize('X', Color.cyan),
                      segments=segments, controller=controller,
                      maxHealth=20 if adaptiveHealth else 50,
                      healthIncrease=2 if adaptiveHealth else 0)
        snakeBoard.addSnake(snake)
        snakeBoard.generatePoint()
        return snakeBoard

    @staticmethod
    def randomAdjacentPosition(pos: Tuple[int, int]) -> Tuple[int, int]:
        direction = Direction.moves()[random.randint(0, len(Direction.moves()) - 1)]
        return pos[0] + direction.value[0], pos[1] + direction.value[1]

    @staticmethod
    def generateSnakeSegments(board, columns: int, rows: int):
        segments = []
        head = (random.randint(0, columns - 1), random.randint(0, rows - 1))
        segments.append(head)
        middle = GeneticTrainer.randomAdjacentPosition(head)
        while not board.inBounds(middle):
            middle = GeneticTrainer.randomAdjacentPosition(head)
        segments.append(middle)
        tail = middle
        while tail in segments or not board.inBounds(tail):
            tail = GeneticTrainer.randomAdjacentPosition(middle)
        segments.append(tail)
        return segments


def main():

    pos = (5, 5)
    posHistory = [pos]
    for i in range(10):
        pos = GeneticTrainer.randomAdjacentPosition(pos)
        posHistory.append(pos)

    from Models.Board import Board
    board = Board()
    for pos in posHistory:
        board.place('X', pos)
    print()
    posSet = set(posHistory)
    print(f"Same length? {len(posHistory) == len(posSet)}")
    if not len(posHistory) == len(posSet):
        print("History:")
        print(posHistory)
        print("\nSet:")
        print(posSet)

    # Ensure that crossover works properly.
    # There should be a mix of 1's, 2's, and potentially a random number or more.
    dict1 = {'weights': torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])}
    dict2 = {'weights': torch.Tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]])}
    crossover = GeneticTrainer.crossover(dict1, dict2, mutation_rate=0.4)
    print(crossover.get('weights'))

    # Randomize the weights
    print(dict1.get('weights'))
    GeneticTrainer.randomize(dict1, mutation_rate=0.5)
    print(dict1.get('weights'))


if __name__ == '__main__':
    main()
