from typing import Tuple
from Models.Board import Board
from Models.Direction import Direction


class SnakeControllable:
    """Defines a standard way to control the next action of a given snake.
    Extend this class and implement `nextDirection` to allow snakes to autonomously move.
    """

    # No argument hints for snake due to circular import issues.
    def nextDirection(self, board: Board, snake, point: Tuple[int, int]) -> Direction:
        """The next move to take given the current state.
        Returns `Direction.none` unless overridden.

        Args:
            board (Board): The current game board state.
            snake (Snake): The snake to decide the next direction.
            point (Tuple[int, int]): The (x, y) position of the point.

        Returns:
            Direction: The next direction to move the given snake.
        """
        return Direction.none

