import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
from tensordiffeq.models import DiscoveryModel
from tensordiffeq.utils import tensor

def f_model(u_model, vars, x, t):
    u = u_model(tf.concat([x,t],1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u,t)
    c1 = vars[0]
    c2 = vars[1]
    f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f_u



lb = np.array([-1.0])
ub = np.array([1.0])

# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)


layer_sizes = [2, 128, 128, 128, 128, 1]
model = DiscoveryModel()

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]
N = X_star.shape[0]
T = t.shape[0]

x = X_star[:, 0:1]
t = X_star[:, 1:2]

X_star = tensor(X_star)


X = [x, t]
print(np.shape(x), np.shape(t), np.shape(X_star))

vars = [tf.Variable(0.0, dtype = tf.float32), tf.Variable(0.0, dtype = tf.float32)]

col_weights = tf.Variable(tf.random.uniform([np.shape(x)[0], 1]))

model.compile(layer_sizes, f_model, X, u_star, vars)

#train loop
model.fit(tf_iter = 10000)

tdq.plotting.plot_weights(model, scale = 10.0)
