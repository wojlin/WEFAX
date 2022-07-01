import sys
import time
import math


def plot_bar(amount: int, max_amount: int, length: int, print_amount: bool = False, label: str = ''):
    percent = amount / max_amount * 100
    LEFT_BRACKET = '\033[0m［'
    RIGHT_BRACKET = '\033[0m］'
    FULL_BLOCK = '\033[0;32m━'
    EMPTY_BLOCK = '\033[0;31m━'
    inner = length - 2
    filled_space = int(math.ceil((percent / 100) * inner))
    left_space = inner - filled_space
    sprint = lambda text: sys.stdout.write(text)
    sprint(f'\r')
    sprint(LEFT_BRACKET)
    sprint(FULL_BLOCK * filled_space)
    if left_space:
        sprint(EMPTY_BLOCK * left_space)
    sprint(RIGHT_BRACKET)
    sprint(' ' + str(int(percent)).rjust(3) + '%')
    if print_amount:
        just = len(str(max_amount))
        sprint(f'     {str(amount).rjust(just)}/{max_amount} ')
    sprint(label)
    sys.stdout.flush()


if __name__ == "__main__":
    for x in range(1000):
        time.sleep(0.01)
        plot_bar(x, 999, 50, True, 'blocks')
