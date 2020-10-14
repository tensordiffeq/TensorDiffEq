from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tensordiffeq as tdq

tf.__version__
@tf.function
def f_model(u, x, t):
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u,t)
    c1 = tdq.constant(.0001)
    c2 = tdq.constant(5.0)
    f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f_u

@tf.function
def u_x_model(u, x, t):
    u_x = tf.gradients(u, x)
    return u, u_x


layer_sizes = [2, 128, 128, 128, 128, 1]
model = tdq.Model()
model.compile(layer_sizes)
