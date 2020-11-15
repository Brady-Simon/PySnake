from typing import List, Tuple
from Models.Direction import Direction
import Controllers.SnakeControllable as Controller


class Snake:

    # Not type hinting controller to prevent circular imports
    def __init__(self, name: str, mark, segments: List[Tuple[int, int]] = None,
                 controller=None, maxHealth: int = None, healthIncrease: int = 0):
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
            healthIncrease (int): The amount to increase `maxHealth` when the snake grows.
        """
        self.name = name
        """The unique name of this snake."""
        self.mark = mark
        """The mark to place on a text based view of a board."""
        self.controller = controller if controller is not None else Controller.SnakeControllable()
        """The controller that can decide the next move of the snake."""
        self.maxHealth: int = maxHealth
        """The maximum number of times the snake can move without starving."""
        self.score: int = 0
        """A representation of how well the snake is doing. Increases as the snake grows."""
        if maxHealth is not None:
            self.health: int = maxHealth
            """The snake's current health. The snake is starving once it reaches 0."""
        self.healthIncrease = healthIncrease
        """The amount of health to add to `maxHealth` each time the snake grows."""

        if segments is None:
            self.segments = [(2, 0), (1, 0), (0, 0)]
        else:
            self.segments = segments
            """The indices of each body part, ordered from head to tail."""

    def head(self) -> (int, int):
        """The position of the snake's head, which is `segments[0]`.
        Returns `None` if the snake has no segments.
        """
        if len(self.segments) == 0:
            return None
        else:
            return self.segments[0]

    def grow(self, size: int = 1):
        """Grows the snake by the given `size`.

        Args:
            size (int): The number of segments to add to the snake.
        """
        if len(self.segments) != 0:
            self.score += 1
            last = self.segments[-1]
            for _ in range(size):
                self.segments.append(last)
            # Reset health if applicable
            if self.maxHealth is not None:
                self.maxHealth += self.healthIncrease
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
