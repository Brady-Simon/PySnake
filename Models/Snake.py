from typing import List, Tuple
from Models.Direction import Direction
import Controllers.SnakeControllable as Controller


class Snake:

    # Not type hinting controller to prevent circular imports
    def __init__(self, name: str, mark, segments: List[Tuple[int, int]] = None, controller=None, maxHealth: int = None):
        """Creates a new snake.

        Args:
            name (str): A unique name for identification.
            mark: The mark to represent the snake on a board.
            segments (List[Tuple[int, int]]): The indices of the snake's body.
                The head is at the first index of the list.
            controller (SnakeControllable): The algorithm that decides how to move the snake.
                Defaults to `SnakeControllable` if left blank, which means no movement.
            maxHealth (int): The number of times the snake can move before starving.
                Getting a point resets this value. Snake never gets hungry if left `None`.
        """
        self.name = name
        self.mark = mark
        self.controller = controller if controller is not None else Controller.SnakeControllable()
        self.maxHealth: int = maxHealth
        if maxHealth is not None:
            self.health: int = maxHealth

        if segments is None:
            self.segments = [(2, 0), (1, 0), (0, 0)]
        else:
            self.segments = segments

    def head(self) -> (int, int):
        if len(self.segments) == 0:
            return None
        else:
            return self.segments[0]

    def grow(self, size: int = 3):
        """Grows the snake by the given `size`.

        Args:
            size (int): The length for which to grow.
        """
        if len(self.segments) != 0:
            last = self.segments[-1]
            for _ in range(size):
                self.segments.append(last)
            # Reset health if applicable
            if self.maxHealth is not None:
                self.health = self.maxHealth

    def move(self, direction: Direction) -> (int, int):
        """Moves the snake in the given `direction`.
        If `direction == Direction.none`, then just returns `head()`.

        Args:
            direction (Direction): The direction to move the snake head.

        Returns:
            (int, int): The (x, y) coordinates of the new head location.
        """
        if direction == Direction.none:
            return self.head()

        # Adjust health if applicable
        if self.maxHealth is not None and self.health > 0:
            # if self.health > 0:
            self.health -= 1

        head = self.head()
        head = (head[0] + direction.value[0], head[1] + direction.value[1])
        self.segments.insert(0, head)
        self.segments.pop(len(self.segments) - 1)



        return head

    @staticmethod
    def adjusted(pos: (int, int), direction: Direction) -> (int, int):
        """Returns the adjusted `pos` by the given `direction`.

        Args:
            pos (int, int): The (x, y) coordinate of the position.
            direction (Direction): The direction to adjust `pos`.

        Returns:
            (int, int): The `pos` adjusted by the given `direction`.
        """
        return pos[0] + direction.value[0], pos[1] + direction.value[1]

    def __contains__(self, item):
        return self.segments.__contains__(item)


def main():
    snake = Snake(name='Player1', mark='X', segments=[(5, 5), (4, 5), (3, 5)])
    print(snake.segments)
    snake.move(Direction.up)
    snake.grow()
    print(snake.segments)
    print((5, 5) in snake)
    print((10, 5) in snake)


if __name__ == '__main__':
    main()
