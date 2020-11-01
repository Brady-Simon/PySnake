from typing import List, Tuple
import random
from Controllers.SnakeControllable import SnakeControllable
from Models.Board import Board
from Models.Snake import Snake
from Models.Direction import Direction


class SnakeAlgorithm(SnakeControllable):

    def nextDirection(self, board: Board, snake: Snake, point: Tuple[int, int]) -> Direction:
        distanceToPoint = (point[0] - snake.head()[0], point[1] - snake.head()[1])
        directions = SnakeAlgorithm.validDirections(board, snake, point)

        if len(directions) == 0:
            allMoves = Direction.moves()
            return allMoves[random.randint(0, len(allMoves) - 1)]
        elif distanceToPoint[0] > 0 and Direction.right in directions:
            return Direction.right
        elif distanceToPoint[0] < 0 and Direction.left in directions:
            return Direction.left
        elif distanceToPoint[1] > 0 and Direction.down in directions:
            return Direction.down
        elif distanceToPoint[1] < 0 and Direction.up in directions:
            return Direction.up
        else:
            return directions[random.randint(0, len(directions) - 1)]

    @staticmethod
    def validDirections(board: Board, snake: Snake, point: Tuple[int, int]) -> List[Direction]:
        return [direction for direction in Direction.moves()
                if board.isEmpty(snake.adjusted(snake.head(), direction))
                or snake.adjusted(snake.head(), direction) == point
                ]


def main():
    board = Board()
    snake = Snake(name='P1', mark='X', segments=[(5, 5), (4, 5), (3, 5)])
    for pos in snake.segments:
        board.place('X', pos)
    ai = SnakeAlgorithm()
    print(ai.validDirections(board, snake, (2, 5)))
    print(ai.nextDirection(board, snake, (2, 5)))
    print(ai.nextDirection(board, snake, (6, 5)))
    print(ai.nextDirection(board, snake, (5, 4)))
    print(ai.nextDirection(board, snake, (5, 6)))


if __name__ == '__main__':
    main()
