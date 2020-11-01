import tkinter as tk
from Controllers.SnakeAlgorithm import SnakeAlgorithm
from Models.SnakeBoard import SnakeBoard
from Models.Direction import Direction
from Models.Snake import Snake
from Models.Color import Color


class SnakeWindow(tk.Frame):

    def __init__(self, master=None, snakeBoard: SnakeBoard = None,
                 humanControllable: bool = True, fps: int = 7,
                 blockSize: int = 50, outlines_enabled: bool = True):

        self.humanControllable = humanControllable
        self.fps = fps
        self.blockSize = max(1, blockSize)
        self.outlines_enabled = outlines_enabled
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
        self.canvas.bind("<Configure>", self.renderScreen)

        if humanControllable:
            self.canvas.bind('<Key>', self.updateDirection)

        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.renderScreen()

        self.after(1000//fps, self.updateGame)

    def renderScreen(self, event=None):
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
        """Returns a string color to use for the block at `(x, y)`."""
        item = self.snakeBoard.board.get((x, y))
        if item == self.snakeBoard.board.empty:
            return 'grey'
        elif (x, y) == self.snakeBoard.point:
            return 'red'
        else:
            return 'deep sky blue'


def main():
    window = SnakeWindow(humanControllable=False, fps=7, blockSize=50, outlines_enabled=True)
    window.mainloop()


if __name__ == '__main__':
    main()
