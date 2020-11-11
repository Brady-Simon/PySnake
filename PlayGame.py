import NeuralNetworkDataParsing as DataParsing
from Views.SnakeWindow import SnakeWindow, generateBoard
from Models.SnakeBoard import SnakeBoard
from NeuralNetwork import NeuralNetwork


def main():
    window = SnakeWindow(humanControllable=False, fps=4, blockSize=50,
                         outlines_enabled=False, using_gradients=True,
                         reset_func=generateBoard, healthBarWidth=10,
                         initial_color=(0, 190, 255), final_color=(255, 0, 255),
                         writeBoardToFile=True)
    window.mainloop()
    # DataParsing.getOutputDataFromFile()


if __name__ == '__main__':
    main()
