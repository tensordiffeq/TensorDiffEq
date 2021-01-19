import numpy as np
import tensorflow as tf
import scipy.io
import math
import tensordiffeq as tdq
import numpy as np
from tensordiffeq.models_new import CollocationSolverND
from tensordiffeq.domains import DomainND
from tensordiffeq.boundaries import *

Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 512)
# Domain.add("y", [-1.0, 1.0], 512)
# Domain.add("z", [-1.0, 1.0], 512)
Domain.add("t", [0.0, 1.0], 100)

N_f = 20000
Domain.generate_collocation_points(N_f)
print(Domain.X_f)

def func_ic(x):
    return np.sin(x * math.pi)


def funcxy_ic(x, y):
    return np.sin(x * math.pi) + np.sin(y * math.pi)


init = IC(Domain, [func_ic], var=[('x')])

upper_x = dirichlectBC(Domain, val=0.0, var='x', target="upper")

lower_x = dirichlectBC(Domain, val=0.0, var='x', target="lower")

BCs = [init, upper_x, lower_x]


def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)

    f_u = u_t + u * u_x - (0.05 / tf.constant(math.pi)) * u_xx

    return f_u


layer_sizes = [2, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)
model.fit(tf_iter = 500, newton_iter = 500)
