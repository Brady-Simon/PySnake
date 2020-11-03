from typing import List, Dict
from Models.Snake import Snake
from Models.Board import Board
from Models.Direction import Direction
from Models.Color import Color


class SnakeBoard:

    def __init__(self, board: Board = None, snakes: List[Snake] = None):
        self.board = board if board is not None else Board()
        if snakes is not None:
            self.snakeDict: Dict[str, Snake] = {snake.name: snake for snake in snakes}
        else:
            self.snakeDict: Dict[str, Snake] = {}
        self.point = None

    def addSnake(self, snake):
        """Adds a snake to the list of snakes."""
        if snake.name in self.snakeDict:
            print(f"Snake {snake.name} already exists.")
        else:
            self.snakeDict[snake.name] = snake
            for pos in snake.segments:
                self.board.place(snake.mark, pos=pos)

    def move(self, snakeName: str, direction: Direction):
        """Moves the snake with the given `snakeName` by `direction`.

        Args:
            snakeName (str): The unique name of the snake.
            direction (Direction): The direction to move the snake.
        """
        if direction == Direction.none:
            return

        snake = self.snakeDict.get(snakeName)
        if snake is not None:
            tail = snake.segments[-1]
            # head = snake.move(direction)
            head = snake.adjusted(snake.head(), direction)
            if head == self.point:
                # Got the point; update snake and generate new point
                snake.move(direction)
                self.board.placeEmpty(tail)
                self.board.place(snake.mark, head)
                snake.grow()
                self.generatePoint()
            elif not self.board.isEmpty(head):
                # Snake is dead :(
                self.remove(snake)
            else:
                # Snake moved in valid direction; update
                snake.move(direction)
                self.board.placeEmpty(tail)
                self.board.place(snake.mark, head)

    def remove(self, snake: Snake):
        """Removes `snake` from `board` and `snakeDict`.
        Previous locations on the board are made empty.
        """
        for segment in set(snake.segments):
            self.board.placeEmpty(segment)
        self.snakeDict.pop(snake.name)

    def update(self):
        """Moves all snakes to their next location."""
        names = list(self.snakeDict.keys())
        for snakeName in names:
            snake = self.snakeDict.get(snakeName)
            self.move(snakeName=snakeName, direction=snake.controller.nextDirection(self.board, snake, self.point))

    def generatePoint(self):
        """Generates a new `point` location."""
        pos, mark = self.board.random()
        # Keep picking a position until you find an empty spot
        while not self.board.isEmpty(pos):
            pos, mark = self.board.random()
        # Update the point position
        self.point = pos
        self.board.place(Color.colorize('0', Color.red), pos)

    def safeDirections(self, snakeName: str) -> List[Direction]:
        """Returns a list of safe directions to move for the snake with `snakeName`."""
        snake = self.snakeDict.get(snakeName)
        return [direction for direction in Direction.moves()
                if self.board.isEmpty(snake.adjusted(snake.head(), direction))
                or snake.adjusted(snake.head(), direction) == self.point
                ]


def main():
    snakeBoard = SnakeBoard()
    snake = Snake(name='Player1', mark=Color.colorize('X', Color.cyan), segments=[(1, 3), (1, 2), (1, 1)])
    snakeBoard.addSnake(snake)
    snakeBoard.generatePoint()
    print(snakeBoard.board)


if __name__ == '__main__':
    main()
