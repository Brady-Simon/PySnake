import tkinter as tk
from random import randint
from typing import Tuple
from Controllers.SnakeAlgorithm import SnakeAlgorithm
from Models.SnakeBoard import SnakeBoard
from Models.Direction import Direction
from Models.Snake import Snake
from Models.Color import Color
import NeuralNetworkDataParsing as DataParsing


class SnakeWindow(tk.Frame):

    def __init__(self, master=None, snakeBoard: SnakeBoard = None,
                 humanControllable: bool = True, fps: int = 7,
                 blockSize: int = 50, outlines_enabled: bool = True,
                 using_gradients: bool = False, reset_func=None,
                 initial_color: Tuple[int, int, int] = (0, 190, 255),
                 final_color: Tuple[int, int, int] = (255, 255, 255),
                 healthBarWidth: int = 5, on_update=None,
                 writeBoardToFile: bool = False):
        """Creates a visual representation of a Snake game.

        Args:
            master: The Tkinter root to use. Optional.
            snakeBoard (SnakeBoard): The board to show on screen. Optional.
            humanControllable (bool): Whether or not a human is controlling the snake.
            fps (int): The number of frames per second to update the game.
            blockSize (int): The width of each block on screen.
            outlines_enabled (bool): Whether or not to include outlines on each block.
            using_gradients (bool): Whether or not to use gradients for the snake.
            reset_func: A function that returns a new `SnakeBoard` upon the game ending.
            initial_color: The color to use for the head of the snake.
            final_color: The color to use for the tail of the snake. Only relevant if `usingGradients` is `True`.
            healthBarWidth (int): The width in pixels for the health bar for hungry snakes.
            on_update: A function that is called after each update and returns the current `SnakeWindow`.
        """

        self.humanControllable = humanControllable
        self.fps = fps
        self.blockSize = max(1, blockSize)
        self.outlines_enabled = outlines_enabled
        self.reset_func = reset_func
        self.using_gradients = using_gradients
        self.initial_color = initial_color
        self.final_color = final_color
        self.on_update = on_update
        self.writeBoardToFile = writeBoardToFile
        if master is None:
            self.master = tk.Tk()
            self.master.title("Slithery Snake")
            self.master.resizable()
        else:
            self.master = master
        super().__init__(master)

        if snakeBoard is None:
            self.snakeBoard = generateBoard()
        else:
            self.snakeBoard = snakeBoard

        self.direction = Direction.up
        self.nextDirection = Direction.up
        self.healthBarWidth = max(0, healthBarWidth)

        # The number of snakes that have a finite amount of health.
        # Add a small bar on the side of the grid to show health.
        hungrySnakeCount = len([snake for snake in self.snakeBoard.snakeDict.values() if snake.maxHealth is not None])

        width = self.snakeBoard.board.columns() * self.blockSize + (self.healthBarWidth * hungrySnakeCount)
        height = self.snakeBoard.board.rows() * self.blockSize
        self.canvas = tk.Canvas(master=self.master, width=width, height=height, highlightthickness=0)
        self.canvas.focus_set()
        self.canvas.bind("<Configure>", lambda _: self.renderScreen())

        if humanControllable:
            self.canvas.bind('<Key>', self.updateDirection)

        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.renderScreen()

        self.after(1000//fps, self.updateGame)

    def renderScreen(self):
        """Renders the contents of `snakeBoard` on screen."""
        self.canvas.delete(tk.ALL)
        # Render a dark grey background behind everything
        # This is useful for a clean background behind the health bar if applicable
        self.canvas.create_rectangle(0, 0, self.canvas.winfo_reqwidth(),
                                     self.canvas.winfo_reqheight(), fill='gray30',
                                     outline='black' if self.outlines_enabled else 'gray30')

        # Render the snake grid
        for row in range(self.snakeBoard.board.rows()):
            for column in range(self.snakeBoard.board.columns()):
                self.canvas.create_rectangle(column * self.blockSize, row * self.blockSize,
                                             (column + 1) * self.blockSize, (row + 1) * self.blockSize,
                                             fill=self.colorFor(column, row),
                                             outline='black' if self.outlines_enabled else self.colorFor(column, row),
                                             )
        # Render the health bars only if some snakes have finite health
        hungrySnakes = [snake for snake in self.snakeBoard.snakeDict.values() if snake.maxHealth is not None]
        if len(hungrySnakes) > 0:
            startingWidth = self.snakeBoard.board.columns() * self.blockSize
            maxHeight = self.snakeBoard.board.rows() * self.blockSize
            for snake in hungrySnakes:
                healthBarHeight = int(maxHeight * snake.health / snake.maxHealth)
                self.canvas.create_rectangle(startingWidth, maxHeight,
                                             startingWidth + self.healthBarWidth, maxHeight - healthBarHeight,
                                             fill='green', outline='black' if self.outlines_enabled else 'gray')
                startingWidth += 5

    def updateDirection(self, event):
        """Updates the local `nextDirection` based on the key-press event."""
        direction = Direction.fromChar(event.char)

        # Write current game state to the file if True
        if self.writeBoardToFile:
            DataParsing.writeToFile(self.snakeBoard, self.direction.toChar())

        if not direction.isOpposite(self.direction) and direction != Direction.none:
            self.nextDirection = direction
            if self.on_update is not None:
                self.on_update(self)

    def updateGame(self):
        """Updates the window with board changes at set `fps` intervals."""

        if self.humanControllable:
            # Update the direction of the snake based on human WASD input.
            self.direction = self.nextDirection
            self.snakeBoard.move(snakeName='P1', direction=self.direction)
        else:
            # Just let the SnakeBoard update itself.
            self.snakeBoard.update()

        # Render any changes on screen.
        self.renderScreen()

        if not self.snakeBoard.isGameOver():
            # Update the game again at a set interval.
            self.after(1000//self.fps, self.updateGame)
        elif self.reset_func is not None:
            # Game ended and reset function is available; reset game.
            self.snakeBoard = self.reset_func()
            self.direction = Direction.up
            self.nextDirection = Direction.up
            self.after(1000 // self.fps, self.updateGame)

    def colorFor(self, x: int, y: int) -> str:
        """Returns a string color to use for the block at `(x, y)`.

        If `using_gradients` is enabled, then the initial color
        will fade to white as the snake gets longer.

        Args:
            x (int): The X-coordinate on the grid.
            y (int): The Y-coordinate on the grid.

        Returns:
            str: The color to fill the block at (x, y). Either a word or hex code.
        """
        item = self.snakeBoard.board.get((x, y))
        if item == self.snakeBoard.board.empty:
            # Empty color
            return 'grey'
        elif (x, y) == self.snakeBoard.point:
            # Point color
            return 'red'
        else:
            # Snake color
            if self.using_gradients:
                availableSnakes = [snake for snake in self.snakeBoard.snakeDict.values() if (x, y) in snake.segments]
                if len(availableSnakes) == 0:
                    return Color.toHex(self.initial_color)
                progress = availableSnakes[0].segments.index((x, y)) / len(availableSnakes[0].segments)
                color = Color.interpolateColor(self.initial_color, self.final_color, progress=progress)
                return Color.rgbToHex(color[0], color[1], color[2])
            else:
                return Color.toHex(self.initial_color)


def generateBoard() -> SnakeBoard:
    """Generates an example `SnakeBoard` to use for training."""
    snakeBoard = SnakeBoard()
    snake = Snake(name='P1', mark='X',
                  segments=[(snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 3),
                            (snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 2),
                            (snakeBoard.board.columns() // 2, snakeBoard.board.rows() - 1)],
                  controller=SnakeAlgorithm(), maxHealth=50)
    snakeBoard.addSnake(snake)
    snakeBoard.generatePoint()
    return snakeBoard


def main():
    window = SnakeWindow(humanControllable=False, fps=1, blockSize=50,
                         outlines_enabled=False, using_gradients=True,
                         reset_func=generateBoard, healthBarWidth=10,
                         initial_color=(0, 190, 255), final_color=(255, 0, 255))
    window.mainloop()


if __name__ == '__main__':
    main()
