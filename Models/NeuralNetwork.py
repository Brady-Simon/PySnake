import torch
import numpy as np
from torch import nn
import NeuralNetworkDataParsing as DataParsing

class NeuralNetwork():

    def __init__(self):
        torch.manual_seed(1)
        # Input Tensor
        self.test_input_data = torch.FloatTensor(DataParsing.getInputDataFromFile())


        #dtype=torch.float)

        # Output Tensor
        self.test_output_data = torch.FloatTensor(DataParsing.getOutputDataFromFile())

        # Learning parameters
        # self.epoch = 1000
        self.learning_rate = 0.01
        self.numberOfInputs = self.test_input_data.shape[1]  # number of features in a dataset/inputlayer
        self.numberOfNeuronsInHiddenLayers = 4  # Does it matter?
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
        # if(True):
        #     print("User tested the following sequence: ")
        convertedToNumpy = testData.numpy()
        convertedToNumpy = convertedToNumpy.astype(np.int32)
        counter = 0
        while counter < len(convertedToNumpy):
            print(convertedToNumpy[counter])
            counter += 1

        predicted_output = self.model(testData)
        convertedToNumpy = predicted_output.detach().numpy()
        print(convertedToNumpy)

        #print("Max value: ", np.argmax(convertedToNumpy))
        return np.argmax(convertedToNumpy)

def main():
    bot = NeuralNetwork()
    bot.trainModel(10000)


if __name__ == '__main__':
    main()
