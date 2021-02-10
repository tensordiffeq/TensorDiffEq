import scipy.io
import math
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
# from tensordiffeq import DomainND
from tensordiffeq.boundaries import *

Domain = DomainND(["x", "y", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 256)
Domain.add("y", [-1.0, 1.0], 256)
Domain.add("t", [0.0, 1.0], 100)

N_f = 20000
Domain.generate_collocation_points(N_f)


def func_ic_xy(x, y):
    return -np.sin(x * math.pi) + -np.sin(y * math.pi)

init = IC(Domain, [func_ic_xy], var=[['x', 'y']])


# Conditions to be considered at the boundaries for the periodic BC
def deriv_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yx = tf.gradients(u_y, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    u_xy = tf.gradients(u_x, y)[0]
    return u, u_x, u_y, u_xx, u_yy, u_xy, u_yx


x_periodic = periodicBC(Domain, ["x", "y"], [deriv_model])


# upper_x = dirichlectBC(Domain, val=0.0, var='x',target="upper")
#
# lower_x = dirichlectBC(Domain, val=0.0, var='x', target="lower")

BCs = [init, x_periodic]


def f_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)

    f_u = u_t + u * u_x - (0.05 / tf.constant(math.pi)) * u_xx

    return f_u


layer_sizes = [3, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)
model.fit(tf_iter=1000, newton_iter=1000)

data = scipy.io.loadmat('burgers_shock.mat')

Exact = data['usol']
Exact_u = np.real(Exact)


x = Domain.domaindict[0]['xlinspace']
t = Domain.domaindict[1]["tlinspace"]

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

u_pred, f_u_pred = model.predict(X_star)

error_u = tdq.find_L2_error(u_pred, u_star)
print('Error u: %e' % (error_u))

U_pred = tdq.get_griddata(X_star, u_pred.flatten(), (X, T))
FU_pred = tdq.get_griddata(X_star, f_u_pred.flatten(), (X, T))

lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

tdq.plotting.plot_solution_domain1D(model, [x, t], ub=ub, lb=lb, Exact_u=Exact_u)
