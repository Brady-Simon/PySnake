import math
from random import randint
import torch
import numpy as np
from torch import nn

from Controllers.SnakeAlgorithm import SnakeAlgorithm
# There is duplicate code in NeuralNetwork.py, this copy is being used by SnakeWindow.py


def writeToFile(snakeBoard, event):
    randomInt = randint(0,6)
    headLoc = snakeBoard.snakeDict.get('P1').head()
    pointLoc = snakeBoard.point
    # if(randomInt == 1):
    print(f"Size: {getLengthOfSnake(snakeBoard)}")
    if(True):
        normalizeSnakeHeadAndFoodLocation(headLoc,pointLoc)
        RawBoardFile = open("RawBoardFile.txt", "a")
        RawBoardFile.write(str(isDirectionSafe(snakeBoard, "up")) + " ")
        RawBoardFile.write(str(isDirectionSafe(snakeBoard, "down")) + " ")
        RawBoardFile.write(str(isDirectionSafe(snakeBoard, "left")) + " ")
        RawBoardFile.write(str(isDirectionSafe(snakeBoard, "right")) + " ")
        RawBoardFile.write(str(getLengthOfSnake(snakeBoard)) + " ")
        RawBoardFile.write(str(normalizeSnakeHeadAndFoodLocation(headLoc,pointLoc)) + "\t\t") ##Writes normalized angle between head and food
        print(interpetEvent(event))
        RawBoardFile.write(interpetEvent(event) + "\n") ##Writes the chosen move
        RawBoardFile.close()
        print(normalizeSnakeHeadAndFoodLocation(snakeBoard.snakeDict.get('P1').head(),snakeBoard.point))

def getInputDataFromFile():
    file = open("RawBoardFile.txt", "r")
    num_lines = sum(1 for line in open('RawBoardFile.txt'))
    inputDataArray = []
    for i in range(num_lines):
        tempArray = []
        isolatedLine = file.readline().strip() #Examines one line at a time
        isolatedInputs = isolatedLine.split("\t")[0] #Parses the single line to contain only the input data
        isolatedSingleInputs = isolatedInputs.split(" ")#Removes the " " and gives access to each input indiviually
        for k in range(len(isolatedSingleInputs)):
            tempArray.append(float(isolatedSingleInputs[k]))
        inputDataArray.append(tempArray)
    return inputDataArray

def getOutputDataFromFile():
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
    return str(snakeboard.board).count("X")/100

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

    normalizedX = pointX-headX
    normalizedY = -(pointY-headY)

    if(normalizedX > 0 and normalizedY == 0):
        return "0.000"
    elif(normalizedX > 0 and normalizedY > 0):
        return "0.125"
    elif(normalizedX == 0 and normalizedY > 0):
        return "0.250"
    elif(normalizedX < 0 and normalizedY > 0):
        return "0.375"
    elif(normalizedX < 0 and normalizedY == 0):
        return "0.500"
    elif(normalizedX <0 and normalizedY < 0):
        return "0.675"
    elif(normalizedX == 0 and normalizedY < 0):
        return "0.750"
    else:
        return "0.875"


def main():
    print(torch.FloatTensor(getInputDataFromFile()))
    print(torch.FloatTensor(getOutputDataFromFile()))


if __name__ == '__main__':
    main()
