
class ProgressBar:

    def __init__(self, bar: str = '█', empty: str = ' '):
        self.bar = bar
        self.empty = empty

    def getProgressBar(self, current, total, length: int = 20):
        progress = min(max(0, current / total), 1)
        barCount = int(progress * length)
        progressBar = '|'
        progressBar += self.bar * barCount
        progressBar += self.empty * (length - barCount)
        progressBar += f'| {current} of {total} ({int(progress * 100)}%)'
        return progressBar

    def printProgress(self, current, total, length: int = 20):
        print('\r' + self.getProgressBar(current, total, length), end='')


def main():
    progressBar = ProgressBar()
    from time import sleep
    maxValue = 20

    print(f"Going from 0 to {maxValue}")
    for i in range(maxValue+1):
        progressBar.printProgress(i, maxValue, length=20)
        sleep(0.5)


if __name__ == '__main__':
    main()
