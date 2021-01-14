import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
import numpy as np
from tensordiffeq.models import CollocationSolver1D
from tensordiffeq.domains import DomainND
from tensordiffeq.boundaries import *

Domain = DomainND(["x", 'y', "t"])

Domain.add("x", [1.0,-1.0], 512)
Domain.add("y", [1.0, -1.0], 512)
Domain.add("t", [0.0,1.0], 100)


upper_x = dirichlectBC(Domain, val = 0.0, var = 'x', time_var = 't', target = "upper")
upper_y = dirichlectBC(Domain, val = 0.0, var = 'y', time_var = 't', target = "upper")
#print(upper_y.create_target_input_repeat())
upper_y.loss()
