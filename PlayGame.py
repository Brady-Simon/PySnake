import NeuralNetworkDataParsing as DataParsing
from Views.SnakeWindow import SnakeWindow, generateBoard


def main():
    #NOTE: This will write new data into RawBoardFile.txt
    # DO NOT DISABLE HUMAN CONTROLLABLE, LEASE AS TRUE
    window = SnakeWindow(humanControllable=True, fps=4, blockSize=50,
                         outlines_enabled=False, using_gradients=True,
                         reset_func=generateBoard, healthBarWidth=10,
                         initial_color=(0, 190, 255), final_color=(255, 0, 255),
                         writeBoardToFile=True)
    window.mainloop()


if __name__ == '__main__':
    main()
