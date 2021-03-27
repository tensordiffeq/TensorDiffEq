from pyfiglet import Figlet
from os import system, name
import sys

def print_screen(model):
    f = Figlet(font='slant')
    print(f.renderText('TensorDiffEq'))
    print("Neural Network Model Summary\n")
    print(model.u_model.summary())