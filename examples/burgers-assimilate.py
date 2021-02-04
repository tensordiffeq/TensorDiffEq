import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
import math
from tensordiffeq.models import CollocationSolver1D


def f_model(u_model, x, t):
    u = u_model(tf.concat([x,t],1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)

    f_u = u_t + u*u_x - (0.05/tf.constant(math.pi))*u_xx

    return f_u



lb = np.array([-1.0])
ub = np.array([1.0])

N0 = 60
N_b = 20 #25 per upper and lower boundary, so 50 total
N_f = 10000
NS = 200

data = scipy.io.loadmat('burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['usol']
Exact_u = Exact

idx_xs = np.random.choice(x.shape[0], NS, replace=False) #need multiple Xs
idx_ts = 75 #for 1 t value
y_s = Exact[idx_xs, idx_ts]
x_s = x[idx_xs,:]
t_s = np.repeat(t[idx_ts,:], len(x_s))

y_s = tf.cast(tf.reshape(y_s, (-1,1)), dtype = tf.float32) #tensors need to be of shape (NS, 1), not (NS, )
x_s = tdq.tensor(x_s)
t_s = tf.cast(tf.reshape(t_s, (-1,1)), dtype = tf.float32)

idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = Exact_u[idx_x,0:1]

u0 = tf.cast(u0, tf.float32)
idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

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



layer_sizes = [2, 128, 128, 128, 128, 1]
model = CollocationSolver1D(assimilate = True)
#model.compile(layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isAdaptive=True, col_weights=col_weights, u_weights=u_weights, g = g)
model.compile(layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, u_lb = u_lb, u_ub = u_ub)
model.compile_data(x_s, t_s, y_s)
#train loops
model.fit(tf_iter = 100, newton_iter =100)

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]

u_pred, f_u_pred = model.predict(X_star)

error_u = tdq.find_L2_error(u_pred, u_star)
print('Error u: %e' % (error_u))


U_pred = tdq.get_griddata(X_star, u_pred.flatten(), (X,T))
FU_pred = tdq.get_griddata(X_star, f_u_pred.flatten(), (X,T))

lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

tdq.plotting.plot_solution_domain1D(model, [x, t],  ub = ub, lb = lb, Exact_u=Exact_u)
