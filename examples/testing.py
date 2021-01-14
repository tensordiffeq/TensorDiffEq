import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
import numpy as np
from tensordiffeq.models import CollocationSolver1D
from tensordiffeq.domains import DomainND
from tensordiffeq.boundaries import *

Domain = DomainND(["x", 'y', 'z', "t"])

Domain.add("x", [1.0,-1.0], 512)
Domain.add("y", [1.0, -1.0], 512)
Domain.add("z", [1.0, -1.0], 512)
Domain.add("t", [0.0,1.0], 100)


upper_x = dirichlectBC(Domain, val = 0.0, var = 'x',target = "upper")
upper_y = dirichlectBC(Domain, val = 0.0, var = 'y',target = "upper")
upper_z = dirichlectBC(Domain, val = 0.0, var = 'z',target = "upper")

#Domain.compile()

#print(upper_y.create_target_input_repeat())
input = upper_z.compile()
input1 = upper_z.compile()
input2 = upper_z.compile()
input3 = upper_z.compile()
input4 = upper_z.compile()
input5 = upper_z.compile()
#print(input)
