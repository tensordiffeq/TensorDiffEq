import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolver1D


lb = np.array([-1.0])
ub = np.array([1.0])

N0 = 60
N_b = 20 #25 per upper and lower boundary, so 50 total
N_f = 10000


data = scipy.io.loadmat('burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['usol']
Exact_u = Exact

idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = Exact_u[idx_x,0:1]

#u0 = tf.cast(u0, tf.float32)
idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# x_f_idx = np.random.choice(x.shape[0], N_f, replace=True)
# t_f_idx = np.random.choice(t.shape[0], N_f, replace=True)

# x_f = tf.convert_to_tensor(x[x_f_idx,:], dtype=tf.float32)
# t_f = tf.convert_to_tensor(t[t_f_idx,:], dtype=tf.float32)
# u_f = tf.convert_to_tensor(Exact_u[x_f_idx, t_f_idx], dtype=tf.float32)

xlimits = np.array([[-1.0, 1.0], [0.0, 1.0]])
X_f = tdq.LatinHypercubeSample(N_f, xlimits)

x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)

X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

x0 = X0[:,0:1]
t0 = X0[:,1:2]

x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)

x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)


t_b_zeros = tf.cast(tf.reshape(np.repeat(0.0, N0),(N0,1)), tf.float32)
u_ub = tf.cast(tf.reshape(np.repeat(0.0, N_b),(N_b,1)), tf.float32)
u_lb = tf.cast(tf.reshape(np.repeat(0.0, N_b),(N_b,1)), tf.float32)

xb0 = tf.concat([x0, x_ub, x_lb], 0)
tb0 = tf.concat([t_b_zeros, t_ub, t_lb],0)
ub0 = tf.concat([u0, u_ub, u_lb],0)

# x0 = xb0
# t0 = tb0
# u0 = ub0


layer_sizes = [2, 128, 128, 128, 128, 1]
model = CollocationSolver1D()
model.compile(layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isAdaptive=True, col_weights=col_weights, u_weights=u_weights, g = g)

#train loops6
model.fit(tf_iter = 1000, newton_iter =10000)
