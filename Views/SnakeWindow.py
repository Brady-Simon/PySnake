import tkinter as tk
from typing import Tuple
from Controllers.SnakeAlgorithm import SnakeAlgorithm
from Models.SnakeBoard import SnakeBoard
from Models.Direction import Direction
from Models.Snake import Snake
from Models.Color import Color


class SnakeWindow(tk.Frame):

    def __init__(self, master=None, snakeBoard: SnakeBoard = None,
                 humanControllable: bool = True, fps: int = 7,
                 blockSize: int = 50, outlines_enabled: bool = True,
                 using_gradients: bool = False,
                 initial_color: Tuple[int, int, int] = (0, 190, 255),
                 final_color: Tuple[int, int, int] = (255, 255, 255)):

        self.humanControllable = humanControllable
        self.fps = fps
        self.blockSize = max(1, blockSize)
        self.outlines_enabled = outlines_enabled
        self.using_gradients = using_gradients
        self.initial_color = initial_color
        self.final_color = final_color
        if master is None:
            self.master = tk.Tk()
            self.master.title("Slithery Snake")
            self.master.resizable()
        else:
            self.master = master
        super().__init__(master)

        if snakeBoard is None:
            self.snakeBoard = SnakeBoard()
            snake = Snake(name='P1', mark=Color.colorize('X', Color.cyan),
                          segments=[(self.snakeBoard.board.columns() // 2, self.snakeBoard.board.rows() - 3),
                                    (self.snakeBoard.board.columns() // 2, self.snakeBoard.board.rows() - 2),
                                    (self.snakeBoard.board.columns() // 2, self.snakeBoard.board.rows() - 1)],
                          controller=SnakeAlgorithm())
            self.snakeBoard.addSnake(snake)
            self.snakeBoard.generatePoint()
        else:
            self.snakeBoard = snakeBoard

        self.direction = Direction.up
        self.nextDirection = Direction.up

        width = self.snakeBoard.board.columns() * self.blockSize
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
        for row in range(self.snakeBoard.board.rows()):
            for column in range(self.snakeBoard.board.columns()):
                self.canvas.create_rectangle(column * self.blockSize, row * self.blockSize,
                                             (column + 1) * self.blockSize, (row + 1) * self.blockSize,
                                             fill=self.colorFor(column, row),
                                             outline='black' if self.outlines_enabled else self.colorFor(column, row),
                                             )

    def updateDirection(self, event):
        """Updates the local `nextDirection` based on the key-press event."""
        direction = Direction.fromChar(event.char)
        if not direction.isOpposite(self.direction) and direction != Direction.none:
            self.nextDirection = direction

    def updateGame(self):
        """Updates the window with board changes at set `fps` intervals."""
        if self.humanControllable:
            self.direction = self.nextDirection
            self.snakeBoard.move(snakeName='P1', direction=self.direction)
        else:
            self.snakeBoard.update()
        self.renderScreen()
        if len(self.snakeBoard.snakeDict) > 0:
            self.after(1000//self.fps, self.updateGame)

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


def main():
    window = SnakeWindow(humanControllable=False, fps=7, blockSize=50,
                         outlines_enabled=True, using_gradients=True,
                         initial_color=(0, 190, 255), final_color=(255, 0, 255))
    window.mainloop()


if __name__ == '__main__':
    main()
