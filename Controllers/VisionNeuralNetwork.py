import torch
import random
from torch import nn
from Models.SnakeBoard import SnakeBoard
from Models.Snake import Snake
from Views.SnakeWindow import SnakeWindow
from Models.Direction import Direction
from Controllers.SnakeControllable import SnakeControllable
from Views.ProgressBar import ProgressBar


class VisionNeuralNetwork(nn.Module, SnakeControllable):

    def __init__(self, snakeName: str = 'P1', file_name: str = "VisionBoardFile.txt"):
        super().__init__()
        torch.manual_seed(1)

        # Learning parameters
        self.snakeName = snakeName
        self.file_name = file_name

        # Input Tensor and output data
        # self.test_input_data = self.getInputDataFromFile()
        # self.test_output_data = self.getOutputDataFromFile()
        # self.numberOfInputs = len(self.test_input_data)
        self.test_input_data = []
        self.test_output_data = []
        self.numberOfInputs = 0

        # Learning weights and algorithm
        self.loss_func = torch.nn.MSELoss()
        self.inputLayer = nn.Linear(36, 24)
        self.hiddenLayer = nn.Linear(24, 12)
        self.outputLayer = nn.Linear(12, 4)

    def forward(self, x):
        x = self.inputLayer(x)
        x = torch.relu(x)

        x = self.hiddenLayer(x)
        x = torch.relu(x)

        x = self.outputLayer(x)
        x = torch.sigmoid(x)
        return x

    def loadTrainingData(self):
        """Loads the input and output data from `self.file_name`."""
        # Input Tensor and output data
        self.test_input_data = self.getInputDataFromFile()
        self.test_output_data = self.getOutputDataFromFile()
        self.numberOfInputs = len(self.test_input_data)

    def train(self, iterations: int = 10000, batch_size: int = 64, learning_rate: float = 0.01, debug: bool = False):

        progressBar = ProgressBar()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        for i in range(iterations):
            # Forward pass: Compute predicted y by passing x to the model
            start = random.randint(0, len(self.test_input_data) - batch_size)
            end = start + batch_size
            model_input = torch.FloatTensor(self.test_input_data[start:end])
            expected_output = torch.FloatTensor(self.test_output_data[start:end])
            predicted_output = self.forward(model_input)

            # Compute and print loss
            loss = self.loss_func(predicted_output, expected_output)
            if debug and (i + 1) % 1000 == 0:
                print('\r' + progressBar.getProgressBar(i + 1, iterations), "{0:.4f} loss".format(loss), end='')

            # Zero gradients and propagate loss backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print an extra line at the end to make sure the progress bar isn't interrupted
        if debug:
            print()

    def getInputDataFromFile(self):
        file = open(self.file_name, "r")
        num_lines = sum(1 for line in open(self.file_name))
        inputDataArray = []
        for i in range(num_lines):
            tempArray = []
            isolatedLine = file.readline().strip()  # Examines one line at a time
            isolatedInputs = isolatedLine.split("-")[0]  # Parses the single line to contain only the input data
            isolatedInputs = isolatedInputs.strip()
            isolatedSingleInputs = isolatedInputs.split(
                " ")  # Removes the " " and gives access to each input individually
            for k in range(len(isolatedSingleInputs)):
                tempArray.append(float(isolatedSingleInputs[k]))
            inputDataArray.append(tempArray)
        # Return the list of inputs
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
        # Return the list of outputs
        return outputDataArray

    def nextDirection(self, snakeBoard: SnakeBoard, snakeName: str) -> Direction:
        """Returns the next move to use given `snakeBoard` and `snakeName`."""
        with torch.no_grad():
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
            tensor: A 36-length tensor that represents the locations of X, O, and empty spaces.
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
    nextDirection = snake_window.nextDirection
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


def offerToSaveModel(model):
    answer = input("Would you like to save this new model? (y/n): ")
    if answer.lower() == 'y':
        torch.save(model, 'vision_neural_network_model')
        print("Model was saved.")


def main():
    file_name = "VisionBoardFile.txt"
    model_file = 'vision_neural_network_model'
    iterations = 20000
    learning_rate = 0.005
    batch_size = 64
    gatheringData = True
    looping = True

    model = VisionNeuralNetwork(file_name=file_name)

    def reset_func():
        return generateBoard(model)

    while looping:
        answer = input("Select an option:\n"
                       + "  1. Create more data\n"
                       + "  2. Load existing model\n"
                       + "  3. Train new model\n"
                       + "<1|2|3>: ")
        if answer == '1':
            # Create more data
            window = SnakeWindow(fps=5, using_gradients=True, healthBarWidth=10, reset_func=reset_func,
                                 on_direction_change=lambda w: update_file(model.file_name, w))
            window.mainloop()
            looping = False
        elif answer == '2':
            # Load existing model
            model = torch.load(model_file)
            model.eval()
            board = generateBoard(model)
            window = SnakeWindow(snakeBoard=board, humanControllable=False, fps=7,
                                 reset_func=reset_func, using_gradients=True)
            window.mainloop()
            looping = False
        elif answer == '3':
            # Train a new model
            model.loadTrainingData()
            model.train(iterations, batch_size=batch_size, learning_rate=learning_rate, debug=True)
            board = generateBoard(model)
            window = SnakeWindow(snakeBoard=board, humanControllable=False, fps=7,
                                 reset_func=reset_func, using_gradients=True)
            window.mainloop()
            offerToSaveModel(model)
            looping = False


if __name__ == '__main__':
    main()
