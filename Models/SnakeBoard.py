from typing import List, Dict, Tuple
from Models.Snake import Snake
from Models.Board import Board
from Models.Direction import Direction
from Models.Color import Color


class SnakeBoard:

    def __init__(self, board: Board = None, snakes: List[Snake] = None):
        self.board = board if board is not None else Board()
        self.snakeDict: Dict[str, Snake] = {}
        self.point = None

        if snakes is not None:
            for snake in snakes:
                self.addSnake(snake)
            self.generatePoint()

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

            if snake.maxHealth is not None and snake.health == 0:
                self.remove(snake)

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
            self.move(snakeName=snakeName, direction=snake.controller.nextDirection(self, snakeName))

    def isGameOver(self):
        """Whether or not the game is over. The game ends when no snakes are left."""
        return len(self.snakeDict) == 0

    def generatePoint(self):
        """Generates a new `point` location."""
        pos, mark = self.board.random()
        # Keep picking a position until you find an empty spot
        while not self.board.isEmpty(pos):
            pos, mark = self.board.random()
        # Update the point position
        self.point = pos
        self.board.place('0', pos)

    def safeDirections(self, snakeName: str) -> List[Direction]:
        """Returns a list of safe directions to move for the snake with `snakeName`."""
        snake = self.snakeDict.get(snakeName)
        return [direction for direction in Direction.moves()
                if self.board.isEmpty(snake.adjusted(snake.head(), direction))
                or snake.adjusted(snake.head(), direction) == self.point
                ]

    def directionFor(self, snakeName: str) -> Direction:
        """Returns the current direction of the snake with `snakeName`.
        Returns `Direction.none` if an error occurs.
        """
        if snakeName not in self.snakeDict.keys():
            return Direction.none
        snake = self.snakeDict.get(snakeName)
        directionValue = (snake.segments[0][0] - snake.segments[1][0],
                          snake.segments[0][1] - snake.segments[1][1])
        # Find directions that match the raw direction value.
        matchingDirections = [direction for direction in Direction.moves() if direction.value == directionValue]
        # Return the first match, or .none if there are not matches (which shouldn't happen).
        if len(matchingDirections) == 0:
            return Direction.none
        else:
            return matchingDirections[0]

    def directionsToPoint(self, pos: Tuple[int, int]) -> List[Direction]:
        """Returns the list of directions to get to the `point` from `pos`."""
        results = []
        delta = (self.point[0] - pos[0], self.point[1] - pos[1])
        results.append(Direction.up) if delta[1] < 0 else results.append(Direction.down)
        results.append(Direction.left) if delta[0] < 0 else results.append(Direction.right)
        return results



# def main():
#     snakeBoard = SnakeBoard()
#     snake = Snake(name='Player1', mark=Color.colorize('X', Color.cyan), segments=[(1, 3), (1, 2), (1, 1)])
#     snakeBoard.addSnake(snake)
#     snakeBoard.generatePoint()
#     print(snakeBoard.board)
#     print(f"Direction: {snakeBoard.directionFor('Player1')}")
#
#
# if __name__ == '__main__':
#     main()
