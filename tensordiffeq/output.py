from pyfiglet import Figlet
from os import system, name
import sys

def print_screen(Domain, model):
    f = Figlet(font='slant')
    print(f.renderText('TensorDiffEq'))