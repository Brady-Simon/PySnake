import torch
import numpy as np
from torch import nn
from random import randint

class NeuralNetwork():

    def __init__(self):
        torch.manual_seed(1)
        # Input Tensor
        self.test_input_data = torch.FloatTensor(self.getInputDataFromFile())


        # Output Tensor
        self.test_output_data = torch.FloatTensor(self.getOutputDataFromFile())

        # Learning parameters
        # self.epoch = 1000
        self.learning_rate = 0.01
        self.numberOfInputs = self.test_input_data.shape[1]  # number of features in a dataset/inputlayer
        self.numberOfNeuronsInHiddenLayers = 6  # Does it matter?
        self.numberOfOutput = 4  # number of neurons in output layer

        # Learning weights and algorithm
        self.parameters = torch.nn.MSELoss()
        self.model = nn.Sequential(nn.Linear(self.numberOfInputs, self.numberOfNeuronsInHiddenLayers),
                                   nn.ReLU(),
                                   nn.Linear(self.numberOfInputs, self.numberOfOutput),
                                   nn.Sigmoid())

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    # The loop of learning!
    def trainModel(self, epoch):
        for i in range(epoch):
            # Forward pass: Compute predicted y by passing x to the model
            predicted_output = self.model(self.test_input_data)

            # Compute and print loss
            loss = self.parameters(predicted_output, self.test_output_data)
            print(f"Iteration: {i}, loss: {loss}")

            # Zero gradients
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            self.optimizer.step()

    def testModel(self, testData):
        convertedToNumpy = testData.numpy()
        convertedToNumpy = convertedToNumpy.astype(np.float)
        counter = 0
        while counter < len(convertedToNumpy):
            print(convertedToNumpy[counter])
            counter += 1

        predicted_output = self.model(testData)
        convertedToNumpy = predicted_output.detach().numpy()
        print(convertedToNumpy)

        #print("Max value: ", np.argmax(convertedToNumpy))
        return np.argmax(convertedToNumpy)

    def writeToFile(self, snakeBoard, event):
        randomInt = randint(0, 6)
        headLoc = snakeBoard.snakeDict.get('P1').head()
        pointLoc = snakeBoard.point
        # if(randomInt == 1):
        print(f"Size: {self.getLengthOfSnake(snakeBoard)}")
        if (True):
            self.normalizeSnakeHeadAndFoodLocation(headLoc, pointLoc)
            RawBoardFile = open("RawBoardFile.txt", "a")
            RawBoardFile.write(str(self.isDirectionSafe(snakeBoard, "up")) + " ")
            RawBoardFile.write(str(self.isDirectionSafe(snakeBoard, "down")) + " ")
            RawBoardFile.write(str(self.isDirectionSafe(snakeBoard, "left")) + " ")
            RawBoardFile.write(str(self.isDirectionSafe(snakeBoard, "right")) + " ")
            RawBoardFile.write(str(self.getLengthOfSnake(snakeBoard)) + " ")
            RawBoardFile.write(str(self.normalizeSnakeHeadAndFoodLocation(headLoc,
                                                                     pointLoc)) + "\t\t")  ##Writes normalized angle between head and food
            print(self.interpetEvent(event))
            RawBoardFile.write(self.interpetEvent(event) + "\n")  ##Writes the chosen move
            RawBoardFile.close()
            print(self.normalizeSnakeHeadAndFoodLocation(snakeBoard.snakeDict.get('P1').head(), snakeBoard.point))

    def getInputDataFromFile(self):
        file = open("RawBoardFile.txt", "r")
        num_lines = sum(1 for line in open('RawBoardFile.txt'))
        inputDataArray = []
        for i in range(num_lines):
            tempArray = []
            isolatedLine = file.readline().strip()  # Examines one line at a time
            isolatedInputs = isolatedLine.split("\t")[0]  # Parses the single line to contain only the input data
            isolatedSingleInputs = isolatedInputs.split(
                " ")  # Removes the " " and gives access to each input indiviually
            for k in range(len(isolatedSingleInputs)):
                tempArray.append(float(isolatedSingleInputs[k]))
            inputDataArray.append(tempArray)
        return inputDataArray

    def getOutputDataFromFile(self):
        file = open("RawBoardFile.txt", "r")
        num_lines = sum(1 for line in open('RawBoardFile.txt'))
        outputDataArray = []
        for i in range(num_lines):
            tempArray = []
            isolatedLine = file.readline().strip()
            isolatedOutputs = isolatedLine.split("\t")[2]
            isolateSingleOutputs = isolatedOutputs.split(" ")
            for k in range(len(isolateSingleOutputs)):
                tempArray.append(float(isolateSingleOutputs[k]))
            outputDataArray.append(tempArray)

        return outputDataArray

    def getLengthOfSnake(snakeboard):
        """returns the normalized length of the snake
        returns int/100"""
        return str(snakeboard.board).count("X") / 100

    def interpetEvent(event):
        if event == 'w':
            return "1 0 0 0\tUp"
        elif event == 's':
            return "0 1 0 0\tDown"
        elif event == 'a':
            return "0 0 1 0\tLeft"
        else:
            return "0 0 0 1\tRight"

    def isDirectionSafe(snakeBoard, direction):
        stringToBeExamined = str(snakeBoard.safeDirections('P1'))
        if direction in stringToBeExamined:
            return 1
        return 0

    def normalizeSnakeHeadAndFoodLocation(headLoc, pointLoc):
        """Finds the angle FROM the head TO the food. Then normalizes the angle to a number between 0 and 1"""
        headX = headLoc[0]
        headY = headLoc[1]
        pointX = pointLoc[0]
        pointY = pointLoc[1]

        normalizedX = pointX - headX
        normalizedY = -(pointY - headY)

        if (normalizedX > 0 and normalizedY == 0):
            return "0.000"
        elif (normalizedX > 0 and normalizedY > 0):
            return "0.125"
        elif (normalizedX == 0 and normalizedY > 0):
            return "0.250"
        elif (normalizedX < 0 and normalizedY > 0):
            return "0.375"
        elif (normalizedX < 0 and normalizedY == 0):
            return "0.500"
        elif (normalizedX < 0 and normalizedY < 0):
            return "0.675"
        elif (normalizedX == 0 and normalizedY < 0):
            return "0.750"
        else:
            return "0.875"

def main():
    bot = NeuralNetwork()
    bot.trainModel(100000)
    testData = torch.tensor([[0, 1, 0, 1, 0.5, 0.375]], dtype=torch.float)
    print(bot.testModel(testData))




if __name__ == '__main__':
    main()
