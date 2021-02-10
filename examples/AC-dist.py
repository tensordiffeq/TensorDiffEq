import numpy as np
import tensorflow as tf
import scipy.io
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolver1D

def f_model(u_model, x, t):
    tf.print(np.shape(x))
    u = u_model(tf.concat([x,t],1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u,t)
    c1 = tdq.utils.constant(.0001)
    c2 = tdq.utils.constant(5.0)
    f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
    return f_u

def u_x_model(u_model, x, t):
    u = u_model(tf.concat([x,t], 1))
    u_x = tf.gradients(u, x)
    return u, u_x


N0 = 200
NS = 200
N_b = 100
N_f = 500000

col_weights = tf.random.uniform([N_f, 1])
u_weights = tf.Variable(100*tf.random.uniform([N0, 1]))

# Grab collocation points using latin hpyercube sampling
xlimits = np.array([[-1.0, 1.0], [0.0, 1.0]])
X_f = tdq.LatinHypercubeSample(N_f, xlimits) #x_f, t_f


lb = np.array([-1.0])
ub = np.array([1.0])

# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_u = np.real(Exact)

#grab training points from domain
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)

idx_xs = np.random.choice(x.shape[0], NS, replace=False) #need multiple Xs
idx_ts = 100 #for 1 t value
y_s = Exact[idx_xs, idx_ts]
x_s = x[idx_xs,:]
t_s = np.repeat(t[idx_ts,:], len(x_s))

y_s = tf.cast(tf.reshape(y_s, (-1,1)), dtype = tf.float32) #tensors need to be of shape (NS, 1), not (NS, )
x_s = tdq.tensor(x_s)
t_s = tf.cast(tf.reshape(t_s, (-1,1)), dtype = tf.float32)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# Grab collocation points using latin hpyercube sampling
x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(np.abs(X_f[:,1:2]), dtype=tf.float32)


X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)


x0 = tf.cast(X0[:,0:1], dtype = tf.float32)
t0 = tf.cast(X0[:,1:2], dtype = tf.float32)

x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)

x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)

layer_sizes = [2, 128, 128, 128, 128, 1]
model = CollocationSolver1D()

def g(lam):
    return lam**2

#model.compile(layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isPeriodic=True, u_x_model=u_x_model, isAdaptive = True, col_weights = col_weights, u_weights = u_weights, g = g, dist = True)
model.compile(layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isPeriodic=True, u_x_model=u_x_model, dist = True)
#train loop
init = model.col_weights
model.fit(tf_iter = 301, newton_iter = 100, batch_sz = 500000)

#generate meshgrid for forward pass of u_pred
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

# print(np.shape(init))
# print(np.shape(model.col_weights))
# plt.scatter(x_f, t_f, c = ((init - model.col_weights)/200))
# print(np.mean(init - model.col_weights))
# plt.show()


tdq.plotting.plot_solution_domain1D(model, [x, t],  ub = ub, lb = lb, Exact_u=Exact_u)

tdq.plotting.plot_weights(model)

tdq.plotting.plot_glam_values(model)

extent = [0.0, 1.0, -1.0, 1.0]
tdq.plotting.plot_residuals(FU_pred, extent)
