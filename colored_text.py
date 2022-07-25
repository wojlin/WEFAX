from datetime import datetime
import sys


class Colors:
    debug = '\033[38;5;249m'
    info = '\033[38;5;255m'
    passed = '\033[38;5;106m'
    error = '\033[38;5;196m'
    warning = '\033[38;5;214m'
    no_color = '\033[0m'


def debug_log(text, color: Colors):
    p_string = '\033[2m'
    p_string += Colors.debug
    p_string += str(datetime.now().strftime('%y-%m-%d %H:%M:%S.%f'))
    p_string += Colors.no_color
    p_string += '   '
    p_string += color
    p_string += str(text)
    p_string += Colors.no_color
    p_string += '\n'
    sys.stdout.write(p_string)
