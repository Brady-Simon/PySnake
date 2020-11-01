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


def main():
    print(Color.colorize("Hello world!", Color.red))
    print(Color.colorize("Hello world!", Color.green))
    print(Color.colorize("Hello world!", Color.blue))


if __name__ == '__main__':
    main()
