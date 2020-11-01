from enum import Enum


class Direction(Enum):
    up = (0, -1)
    right = (1, 0)
    down = (0, 1)
    left = (-1, 0)
    none = (0, 0)

    @staticmethod
    def moves():
        """Returns a list of all directions except `none`."""
        return [Direction.up, Direction.down, Direction.left, Direction.right]

    def isOpposite(self, direction) -> bool:
        """Whether or not `direction` is on the opposite side of `self`."""
        return self.value[0] + direction.value[0] == 0 and self.value[1] + direction.value[1] == 0

    @staticmethod
    def fromChar(char: str):
        char = char.lower()
        if char.lower() == 'w':
            return Direction.up
        elif char.lower() == 'a':
            return Direction.left
        elif char.lower() == 's':
            return Direction.down
        elif char.lower() == 'd':
            return Direction.right
        else:
            return Direction.none
