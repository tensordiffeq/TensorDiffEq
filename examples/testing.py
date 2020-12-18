import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
import numpy as np
from tensordiffeq.models import CollocationSolver1D
from tensordiffeq.domains import DomainND

Domain = DomainND([[-1,1], [0,1]], [512, 100])

Domain.add("x", [1,-1], 512)
Domain.add("t", [0,1], 100)

print(Domain.domaindict)
