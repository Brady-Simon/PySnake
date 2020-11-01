import random


class Board:

    def __init__(self, dim: (int, int) = (10, 10), empty='•'):
        """Creates a board with the given (x, y) dimensions `dim`.

        Args:
            dim (int, int): The dimensions of the board (x, y).
        """
        self.dim = dim
        self.empty = empty
        self.board = []
        # Rows
        for _ in range(dim[1]):
            # Columns
            self.board.append([empty for _ in range(dim[0])])

    def columns(self):
        """The number of columns on the board/"""
        return self.dim[0]

    def rows(self):
        """The number of rows on the board/"""
        return self.dim[1]

    def get(self, pos: (int, int)):
        """Gets the mark at the (x, y) `pos`."""
        return self.board[pos[1]][pos[0]]

    def place(self, mark, pos: (int, int)):
        """Places the given `mark` at `pos`.

        Args:
            mark: The item to place at `pos`.
            pos (int, int): The (x, y) coordinates to place the `mark`.
        """
        self.board[pos[1]][pos[0]] = mark

    def placeEmpty(self, pos: (int, int)):
        """Places the `self.empty` mark at `pos`."""
        self.board[pos[1]][pos[0]] = self.empty

    def inBounds(self, pos: (int, int)) -> bool:
        """Whether or not `pos` is in the bounds of `self.dim`"""
        return 0 <= pos[0] < self.dim[0] and 0 <= pos[1] < self.dim[1]

    def isEmpty(self, pos: (int, int)) -> bool:
        """Whether or not the item at the given `pos` is empty.

        Args:
            pos (int, int): The (x, y) coordinates to check.

        Returns:
            bool: True if the mark at `pos` is equal to `self.empty`.
        """
        if not self.inBounds(pos):
            return False
        else:
            return self.get(pos) == self.empty

    def random(self) -> ((int, int), object):
        """Returns a random position and the item at the position."""
        randX = random.randint(0, self.columns()-1)
        randY = random.randint(0, self.rows() - 1)
        item = self.get((randX, randY))
        return (randX, randY), item

    def __str__(self):
        boardString = ''
        for row in range(self.rows()):
            for col in range(self.columns()):
                boardString += f' {self.get((col, row))} '
            boardString += '\n'
        return boardString


def main():
    board = Board(dim=(7, 5), empty='•')
    board.place('X', (1, 2))
    print(board.get((1, 2)))
    print(board)


if __name__ == '__main__':
    main()
