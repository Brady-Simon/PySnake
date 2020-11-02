from typing import Tuple


class Color:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'
    magenta = '\033[35m'
    cyan = '\033[36m'
    white = '\033[37m'
    underline = '\033[4m'
    reset = '\033[0m'

    @staticmethod
    def colorize(text: str, color: str) -> str:
        """Colorizes the plain `text` with the given `color` code."""
        reset = '\033[0m'
        return color + text + reset

    @staticmethod
    def toHex(color: Tuple[int, int, int]):
        """Converts the given RGB color tuple into hex format, #RRGGBB."""
        return Color.rgbToHex(color[0], color[1], color[2])

    @staticmethod
    def rgbToHex(r: int, g: int, b: int) -> str:
        """Converts the given RGB values (0-255) into hex format, #RRGGBB.

        Args:
            r (int): The amount of red, between 0 and 255.
            g (int): The amount of green, between 0 and 255.
            b (int): The amount of blue, between 0 and 255.

        Returns:
            str: The string representation of the hex color (#RRGGBB).
        """
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    @staticmethod
    def lerp(initial, final, progress):
        """Lerps between `initial` and `final` based on `progress`."""
        return initial * (1 - progress) + final * progress

    @staticmethod
    def interpolateColor(initialColor, finalColor, progress: float) -> (int, int, int):
        """Interpolates between `initialColor` and `finalColor`.

        Args:
            initialColor (int, int, int): The starting RGB color.
            finalColor (int, int, int): The ending RGB color.
            progress (float): The progress between colors (0.0-1.0).

        Returns:
            (int, int, int): The interpolated RGB color.
        """
        red = Color.clamp(Color.lerp(initialColor[0], finalColor[0], progress))
        green = Color.clamp(Color.lerp(initialColor[1], finalColor[1], progress))
        blue = Color.clamp(Color.lerp(initialColor[2], finalColor[2], progress))
        return red, green, blue

    @staticmethod
    def clamp(n, smallest: int = 0, largest: int = 255) -> int:
        """Clamps `n` between `smallest` and `largest`."""
        return int(max(smallest, min(n, largest)))


def main():
    print(Color.colorize("Hello world!", Color.red))
    print(Color.colorize("Hello world!", Color.green))
    print(Color.colorize("Hello world!", Color.blue))


if __name__ == '__main__':
    main()
