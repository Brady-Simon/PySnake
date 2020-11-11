import torch
import numpy as np
from torch import nn
from random import randint
from Models.SnakeBoard import SnakeBoard
from Models.Snake import Snake
from Views.SnakeWindow import SnakeWindow
from Models.Direction import Direction
from Controllers.SnakeControllable import SnakeControllable


class VisionNeuralNetwork(nn.Module, SnakeControllable):

    def __init__(self, learning_rate=0.01, snakeName: str = 'P1', file_name: str = "VisionBoardFile.txt"):
        super().__init__()
        torch.manual_seed(1)

        # Learning parameters
        # self.epoch = 1000
        self.learning_rate = learning_rate
        self.snakeName = snakeName
        self.file_name = file_name

        # Input Tensor and output data
        self.test_input_data = torch.FloatTensor(self.getInputDataFromFile())
        self.test_output_data = torch.FloatTensor(self.getOutputDataFromFile())
        self.numberOfInputs = self.test_input_data.shape[1]

        # Learning weights and algorithm
        self.loss_func = torch.nn.MSELoss()
        self.inputLayer = nn.Linear(32, 20)
        self.hiddenLayer = nn.Linear(20, 12)
        self.outputLayer = nn.Linear(12, 4)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = self.inputLayer(x)
        x = torch.relu(x)

        x = self.hiddenLayer(x)
        x = torch.relu(x)

        x = self.outputLayer(x)
        x = torch.sigmoid(x)
        return x

    def train(self, iterations: int = 10000, debug: bool = False):
        for i in range(iterations):
            # Forward pass: Compute predicted y by passing x to the model
            predicted_output = self.forward(self.test_input_data)

            # Compute and print loss
            loss = self.loss_func(predicted_output, self.test_output_data)
            if debug:
                print(f"Iteration: {i}, loss: {loss}")

            # Zero gradients and propagate loss backwards
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # def writeToFile(self, snakeBoard: SnakeBoard, nextDirection: Direction):
    #     """Writes the board and its predicted output to the file."""
    #
    #     tensor = self.boardToTensor(snakeBoard, self.snakeName)
    #     file = open(self.file_name, "a")
    #
    #     for point in tensor:
    #         file.write(f"{point} ")
    #     file.write("- ")  # Separator between input/output
    #     for direction in Direction.moves():
    #         file.write(f"{1.0 if direction == nextDirection else 0.0 }")
    #
    #     file.close()

    def getInputDataFromFile(self):
        file = open(self.file_name, "r")
        num_lines = sum(1 for line in open(self.file_name))
        inputDataArray = []
        for i in range(num_lines):
            tempArray = []
            isolatedLine = file.readline().strip()  # Examines one line at a time
            isolatedInputs = isolatedLine.split("-")[0]  # Parses the single line to contain only the input data
            isolatedInputs = isolatedInputs.strip()
            print(isolatedInputs)
            isolatedSingleInputs = isolatedInputs.split(
                " ")  # Removes the " " and gives access to each input individually
            for k in range(len(isolatedSingleInputs)):
                tempArray.append(float(isolatedSingleInputs[k]))
            inputDataArray.append(tempArray)
        return inputDataArray

    def getOutputDataFromFile(self):
        file = open(self.file_name, "r")
        num_lines = sum(1 for line in open(self.file_name))
        outputDataArray = []
        for i in range(num_lines):
            tempArray = []
            isolatedLine = file.readline().strip()
            isolatedOutputs = isolatedLine.split("-")[1]
            isolatedOutputs = isolatedOutputs.strip()
            isolateSingleOutputs = isolatedOutputs.split(" ")
            for k in range(len(isolateSingleOutputs)):
                tempArray.append(float(isolateSingleOutputs[k]))
            outputDataArray.append(tempArray)

        return outputDataArray

    def nextDirection(self, snakeBoard: SnakeBoard, snakeName: str) -> Direction:
        """Returns the next move to use given `snakeBoard` and `snakeName`."""
        tensor = self.boardToTensor(snakeBoard, snakeName)
        tensorResult = self.forward(tensor)
        argmax = torch.argmax(tensorResult).item()
        return Direction.moves()[argmax]

    @staticmethod
    def boardToTensor(gameBoard: SnakeBoard, snakeName: str):
        """Converts a board to a usable Tensor for input into the ANN.

        Args:
            gameBoard (TicTacToeBoard): The board being converted to a tensor.
            snakeName (str): The name of the snake to control.

        Returns:
            tensor: A 27-length tensor that represents the locations of X, O, and empty spaces.
        """
        result = []
        # Add the snake's current direction
        result.extend([1.0 if gameBoard.directionFor(snakeName) == direction
                       else 0.0 for direction in Direction.moves()])
        # Add the direction to the point
        directionsToPoint = gameBoard.directionsToPoint(gameBoard.snakeDict.get(snakeName).head())
        result.extend([1.0 if direction in directionsToPoint else 0.0 for direction in Direction.moves()])
        # Add the vision around snake
        visionDeltas = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        head = gameBoard.snakeDict.get(snakeName).head()
        for delta in visionDeltas:
            result.extend(VisionNeuralNetwork.visionTo(gameBoard, head, delta))
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


def generateBoard(controller, name='P1', mark='X') -> SnakeBoard:
    """Generates an example `SnakeBoard` to use for training."""
    snakeBoard = SnakeBoard()
    snake = Snake(name=name, mark=mark,
                  segments=[(snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 3),
                            (snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 2),
                            (snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 1)],
                  controller=controller, maxHealth=50)
    snakeBoard.addSnake(snake)
    snakeBoard.generatePoint()
    return snakeBoard


def update_file(file_name: str, snake_window: SnakeWindow):
    """Writes the board and its predicted output to the file."""
    nextDirection = snake_window.direction
    board = snake_window.snakeBoard

    tensor = VisionNeuralNetwork.boardToTensor(board, 'P1')
    file = open(file_name, "a")

    # Writes the inputs
    for point in tensor:
        file.write(f"{point} ")
    # Separator between input/output
    file.write("-")
    # Write the outputs
    for direction in Direction.moves():
        file.write(f" {1.0 if direction == nextDirection else 0.0}")
    # Close the file
    file.write("\n")
    file.close()


def main():

    file_name = "VisionBoardFile.txt"
    iterations = 10000
    trainingModel = True
    looping = True

    while looping:
        answer = input("Are you trying to load more data? (y/n): ")
        if answer.lower() == 'y':
            trainingModel = True
            looping = False
        elif answer.lower() == 'n':
            trainingModel = False
            looping = False
        else:
            print("Input not recognized.")

    model = VisionNeuralNetwork()
    if trainingModel:
        # Load the snake window and add data as it comes
        model = VisionNeuralNetwork()
        window = SnakeWindow(fps=5, using_gradients=True,
                             healthBarWidth=10,
                             on_update=lambda w: update_file(model.file_name, w))
        window.mainloop()
    else:
        # Train the model and play the game
        model.train(iterations)
        board = generateBoard(model)
        reset_func = lambda: generateBoard(model)
        window = SnakeWindow(snakeBoard=board, humanControllable=False, fps=7,
                             reset_func=reset_func, using_gradients=True)
        window.mainloop()


if __name__ == '__main__':
    main()
