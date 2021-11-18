import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
from tensordiffeq.models import DiscoveryModel
from tensordiffeq.utils import tensor
import pickle

#####################
## Discovery Model ##
#####################


# Put params into a list
params = [tf.Variable(0.0001, dtype=tf.float32), tf.Variable(0.0001, dtype=tf.float32)]


# Define f_model, note the `vars` argument. Inputs must follow this order!
# Define f_model, note the `vars` argument. Inputs must follow this order!
def f_model(u_model, var, x, t):
    # keep track of our gradients
    g1 = var[0]
    g2 = var[1]

    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_t = tf.gradients(u, t)
    u_xx = tf.gradients(u_x, x)

    tmp = g1 * (u ** 3 - u) - g2 * u_xx
    tmp_x = tf.gradients(tmp, x)[0]
    tmp_xx = tf.gradients(tmp_x, x)[0]

    f_u = u_t - tmp_xx
    return f_u

# Import data, same data as Raissi et al

with open('CH.pkl', 'rb') as f:
    data = pickle.load(f)

Exact_u = data

x = np.linspace(-1, 1, np.shape(data)[1])
t = np.linspace(0, 1, np.shape(data)[0])

# generate all combinations of x and t
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = tensor(Exact_u.T.flatten()[:, None])

x = X_star[:, 0:1]
t = X_star[:, 1:2]

# append to a list for input to model.fit
X = [x, t]

# define col_weights for SA discovery model, can be removed
col_weights = tf.Variable(tf.random.uniform([np.shape(x)[0], 1]))

# define MLP depth and layer width
layer_sizes = [2, 128, 128, 128, 128, 1]

# initialize, compile, train model
model = DiscoveryModel()
model.compile(layer_sizes, f_model, X, u_star, params)  # baseline discovery approach can be done by simply removing the col_weights arg

# train loop
model.fit(tf_iter=10000)

# doesnt work quite yet
tdq.plotting.plot_weights(model, scale=10.0)
