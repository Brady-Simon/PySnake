import torch
import torch.nn as nn
from Controllers.SnakeControllable import SnakeControllable
from Models.SnakeBoard import SnakeBoard
from Models.Snake import Snake
from Models.Direction import Direction
from Controllers.VisionNeuralNetwork import VisionNeuralNetwork

import matplotlib
matplotlib.use("TkAgg")  # Using TkAgg to prevent issues with Tkinter SnakeWindow
from matplotlib import pyplot as plt


class GenericSnakeAI(nn.Module, SnakeControllable):
    """A neural network that decides snake movements based on directions.

    Input lists should match the following structure:
     - Size X list
     - Direction of the point [0, 1, 1, 0] (length (4)
     - Direction snake is facing [1, 0, 0, 0] (length 4)
     - Vision around the snake (3x3 grid, 5x5 grid, etc.)

     The output from `forward()` is a 4-length tensor indicating the directions.
    """

    def __init__(self, vision_radius: int = 2, debug: bool = False):
        """Creates a `GenericSnakeAI`.

        Args:
            vision_radius (int): How far the snake can see in blocks.
            debug (bool): Whether or not to include diagnostic print statements.
        """
        super().__init__()
        self.vision_radius = vision_radius
        self.debug = debug
        self.lossHistory = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inputSize: int = 36
        hiddenSize1 = 24
        hiddenSize2 = 12
        outputSize = 4
        self.inputLayer = nn.Linear(inputSize, hiddenSize1).to(self.device)
        self.hiddenLayer = nn.Linear(hiddenSize1, hiddenSize2).to(self.device)
        self.outputLayer = nn.Linear(hiddenSize2, outputSize).to(self.device)

    def forward(self, x):
        """Forwards tensor `x` through the neural network.

        Args:
            x (tensor): The tensor to forward.

        Returns:
            tensor: The value generated by the neural network.
        """
        x = self.inputLayer(x)
        x = torch.relu(x)

        x = self.hiddenLayer(x)
        x = torch.relu(x)

        x = self.outputLayer(x)
        x = torch.sigmoid(x)
        return x

    def nextDirection(self, snakeBoard: SnakeBoard, snakeName: str) -> Direction:
        """Returns the next move to use given `snakeBoard` and `snakeName`."""
        tensor = self.boardToTensor(snakeBoard, snakeName)
        tensorResult = self.forward(tensor)
        argmax = torch.argmax(tensorResult).item()
        return Direction.moves()[argmax]

    def boardToTensor(self, gameBoard: SnakeBoard, snakeName: str):
        """Converts a board to a usable Tensor for input into the ANN.

        Args:
            gameBoard (TicTacToeBoard): The board being converted to a tensor.
            snakeName (str): The name of the snake to control.

        Returns:
            tensor: A 36-length tensor.
        """
        result = []
        # Add the snake's current direction (4)
        result.extend([1.0 if gameBoard.directionFor(snakeName) == direction
                       else 0.0 for direction in Direction.moves()])
        # Add the safe directions (4)
        safeDirections = gameBoard.safeDirections(snakeName)
        result.extend([1.0 if direction in safeDirections else 0.0 for direction in Direction.moves()])
        # Add the direction to the point (4)
        directionsToPoint = gameBoard.directionsToPoint(gameBoard.snakeDict.get(snakeName).head())
        result.extend([1.0 if direction in directionsToPoint else 0.0 for direction in Direction.moves()])
        # Add the vision around snake (8x3=24)
        visionDeltas = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        head = gameBoard.snakeDict.get(snakeName).head()
        for delta in visionDeltas:
            result.extend(self.visionTo(gameBoard, head, delta))
        return torch.Tensor(result)

    @staticmethod
    def visionTo(snakeBoard, head, delta) -> list:
        result = [0.0, 0.0, 0.0]  # Food, body, empty/wall
        pos = head
        distance = 0
        while snakeBoard.board.inBounds(pos):
            pos = (pos[0] + delta[0], pos[1] + delta[1])
            distance += 1
            if snakeBoard.point == pos:
                result[0] = 1.0
            elif not snakeBoard.board.isEmpty:
                result[1] = 1.0
            # elif snakeBoard.board.isEmpty(pos):
            #     result[2] = 1.0
        result[2] = 1 / distance
        return result


def main():
    model = GenericSnakeAI()

    vision_model: VisionNeuralNetwork = torch.load('vision_neural_network_model')
    initial_state_dict = model.state_dict()
    # initial_state_dict = vision_model.state_dict().copy()
    # model.load_state_dict(initial_state_dict)

    from Controllers.Genetics.GeneticTrainer import GeneticTrainer

    def get_model():
        return GenericSnakeAI()

    # state_dict, fitness_history = GeneticTrainer.train(model, population=256, generations=32,
    #                                                    workers=8, mutation_rate=0.05)

    state_dict, fitness_history = GeneticTrainer.startSimulation(get_model, initial_state_dict=initial_state_dict,
                                                                 population=50, generations=50, mutation_rate=0.005)
    model.load_state_dict(state_dict)

    figure = plt.gcf()
    figure.canvas.set_window_title("Genetic Training Results")
    plt.title(f"Fitness History")
    plt.grid(axis='y')
    plt.ylabel("Max Fitness")
    plt.xlabel("Generation")
    plt.plot(fitness_history)
    plt.show()

    from Views.SnakeWindow import SnakeWindow

    def get_board():
        snakeBoard = SnakeBoard()
        segments = GeneticTrainer.generateSnakeSegments(snakeBoard.board,
                                                        snakeBoard.board.columns(),
                                                        snakeBoard.board.rows())
        snake = Snake(name='AI', mark='X',
                      segments=segments,
                      controller=model, maxHealth=50)
        snakeBoard.addSnake(snake)
        snakeBoard.generatePoint()
        return snakeBoard

    board = get_board()
    window = SnakeWindow(snakeBoard=board, humanControllable=False, fps=7, reset_func=get_board)
    window.mainloop()

    # Ask to save the trained model.
    while True:
        shouldSaveModel = input("Would you like to save the model? (y/n): ")
        if shouldSaveModel.lower() == 'y':
            torch.save(state_dict, 'genetic_state_dict')
            break
        elif shouldSaveModel.lower() == 'n':
            print("Exiting...")
            break
        else:
            print("Input not recognized.")


if __name__ == '__main__':
    main()
