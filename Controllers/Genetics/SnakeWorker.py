import torch
import random
from Models.SnakeBoard import SnakeBoard
from Models.Snake import Snake
from Models.Color import Color
# from Controllers.GenericSnakeAI import GenericSnakeAI


class SnakeWorker:

    @staticmethod
    def simulate(model, steps: int = 1024, randomizer_step: int = 5, mutation_rate=0.05) -> (float, dict):
        """Simulates several games of Snake from `board` using `model`.

        Args:
            model: The PyTorch model to use. Must also conform to SnakeControllable.
            steps (int): The number of games to play.
            randomizer_step (int): How often to randomize the model weights.
            mutation_rate (float): The mutation rate of the genes.
                Higher indicates more mutation. Usually in the range of 0...1.

        Returns:
            (float, dict): The max fitness for the given simulation and the accompanying state dictionary.
        """
        with torch.no_grad():
            # Copy the state dict so that other instances are not effected.
            state_dict = model.state_dict().copy()
            best_state_dict = state_dict.copy()
            max_fitness = 0
            for step in range(steps):
                move_counter = 0
                current_score = 0
                snakeBoard = SnakeWorker.generateDefaultBoard(controller=model)

                # Play the game until the snake dies
                while not snakeBoard.isGameOver():
                    move_counter += 1
                    current_score = snakeBoard.snakeDict.get('AI').score
                    snakeBoard.update()

                # Update the max fitness if applicable
                fitness = SnakeWorker.fitness(current_score, move_counter)
                if max_fitness < fitness:
                    # Snake was better; update values
                    max_fitness = fitness
                    best_state_dict = state_dict.copy()

                # Randomize the state dict at regular intervals
                if (step + 1) % randomizer_step == 0:
                    SnakeWorker.randomize(state_dict, mutation_rate)
                    model.load_state_dict(state_dict)

        return max_fitness, best_state_dict

    @staticmethod
    def randomize(state_dict, mutation_rate: float):
        """Randomizes the `state_dict` randomly based on `percent_variation`."""
        for key in state_dict.keys():
            with torch.no_grad():
                # Flatten mask and then reshape it
                matrix = state_dict[key]
                mask = torch.flatten(torch.ones(matrix.size()))
                mask = torch.Tensor([2 * random.random() - 1
                                     if random.random() < mutation_rate
                                     else num for num in mask])
                mask = mask.reshape(matrix.size())
                matrix *= mask
                state_dict[key] = matrix

    @staticmethod
    def fitness(score: int, turns: int) -> float:
        """Returns the snake fitness based on the moves given."""
        return 1.8 ** score + 1.05 ** turns

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
    # Test and make sure the functions work
    stateDict = {
        'weights': torch.Tensor([[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]]),
        'biases': torch.Tensor([[3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3]]),
    }
    worker = SnakeWorker()
    worker.randomize(stateDict, mutation_rate=0.5)
    print(stateDict)


if __name__ == '__main__':
    main()
