from pyfiglet import Figlet
from os import system, name
import sys

def print_screen(model, discovery_model=False):
    f = Figlet(font='slant')
    print(f.renderText('TensorDiffEq'))
    if discovery_model:
        print("Running Discovery Model for Parameter Estimation\n\n")
    print("Neural Network Model Summary\n")
    print(model.u_model.summary())