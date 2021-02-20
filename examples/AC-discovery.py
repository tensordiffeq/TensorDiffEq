import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
from tensordiffeq.models import DiscoveryModel
from tensordiffeq.utils import tensor

#####################
## Discovery Model ##
#####################


# Put params into a list
params = [tf.Variable(0.0, dtype=tf.float32), tf.Variable(0.0, dtype=tf.float32)]


# Define f_model, note the `vars` argument. Inputs must follow this order!
def f_model(u_model, var, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    c1 = var[0]  # tunable param 1
    c2 = var[1]  # tunable param 2
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u


# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = data['uu']
Exact_u = np.real(Exact)

# generate all combinations of x and t
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

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
model.compile(layer_sizes, f_model, X, u_star, params,
              col_weights=col_weights)  # baseline discovery approach can be done by simply removing the col_weights arg

# an example as to how one could modify an optimizer, in this case the col_weights optimizer
model.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005,
                                                      beta_1=.95)

# train loop
model.fit(tf_iter=10000)

# doesnt work quite yet
tdq.plotting.plot_weights(model, scale=10.0)
