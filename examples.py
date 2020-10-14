from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smt.sampling_methods import LHS

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
model = tdq.CollocationModel1D()



N0 = 200
N_b = 100
N_f = 20000

col_weights = tf.Variable(tf.random.uniform([N_f, 1]))
u_weights = tf.Variable(100*tf.random.uniform([N0, 1]))

# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)

#grab training points from domain
idx_x = np.random.choice(x.shape[0], N0, replace=False)

u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# Grab collocation points using latin hpyercube sampling

X_f = LatinHypercubeSample(N_f, xlimits) #x_f, t_f

IVs = x[idx_x,:]

lb = np.array([-1.0])
ub = np.array([1.0])

initial = np.concatenate((x0, 0*x0), 1)
lower = np.concatenate((0*tb + lb[0], tb), 1)
upper = np.concatenate((0*tb + ub[0], tb), 1)

X0 = tdq.tensor(initial) # (x0, 0)
X_lb = tdq.tensor(lower) # (lb[0], tb)
X_ub =  tdq.tensor(upper) # (ub[0], tb)

model.compile()


model.compile(layer_sizes)
