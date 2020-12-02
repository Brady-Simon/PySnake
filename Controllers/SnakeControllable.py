from Models.Direction import Direction


class SnakeControllable:
    """Defines a standard way to control the next action of a given snake.
    Extend this class and implement `nextDirection` to allow snakes to autonomously move.
    """

    def nextDirection(self, snakeBoard, snakeName: str) -> Direction:
        """The next move to take given the current state.
                Returns `Direction.none` unless overridden.

                Args:
                    snakeBoard (SnakeBoard): The current Snake board.
                    snakeName (str): The snake to find in `snakeBoard`.

                Returns:
                    Direction: The next direction to move the given snake.
                """
        return Direction.none
