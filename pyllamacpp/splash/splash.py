import os.path
import os

COLOR = 32

def splash():
    SPLASH_FILE = os.path.join(os.path.dirname(__file__), 'splash.txt')
    with open(SPLASH_FILE, 'r') as file:
        file_lines = file.readlines()
        max_length = max([len(line) for line in file_lines])

        size = os.get_terminal_size()

        if size.columns > max_length:
            lines = [line.replace('\n', '') for line in file_lines]
        else:
            lines = '''
          PyLLaMA.cpp
Created by Diego Morales Román
                         2023
            '''.splitlines()


        max_length = max([len(line) for line in lines])
        padding = ' ' * ((size.columns - max_length) // 2)

        print(f'\033[{COLOR}m')
        for line in lines:
            print(padding + line)
        print('\033[0m', end='')
