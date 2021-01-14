import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
import numpy as np
from tensordiffeq.models_new import CollocationSolver1D
from tensordiffeq.domains import DomainND
from tensordiffeq.boundaries import *

Domain = DomainND(["x", 'y', 'z', "t"])

Domain.add("x", [-1.0,1.0], 512)
Domain.add("t", [0.0,1.0], 100)


upper_x = dirichlectBC(Domain, val = 0.0, var = 'x',target = "upper")
lower_x = dirichlectBC(Domain, val = 0.0, var = 'x',target = "lower")



print(upper_x.out)




def f_model(u_model, inputs):
    u = u_model(tf.concat([x,t],1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)

    f_u = u_t + u*u_x - (0.05/tf.constant(math.pi))*u_xx

    return f_u
