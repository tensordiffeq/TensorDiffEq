import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensordiffeq as tdq
from tensordiffeq.boundaries import *
from tensordiffeq.models import CollocationSolverND
from tensorflow.math import sin
from tensordiffeq.utils import constant

Domain = DomainND(["x", "y"])

Domain.add("x", [-1.0, 1.0], 1001)
Domain.add("y", [-1.0, 1.0], 1001)

N_f = 10000
Domain.generate_collocation_points(N_f)


def f_model(u_model, x, y):
    u = u_model(tf.concat([x, y], 1))
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    a1 = constant(1.0)
    a2 = constant(4.0)
    ksq = constant(1.0)
    pi = constant(math.pi)

    # we use this specific forcing term because we have an exact analytical solution for this case
    # to compare the results of the PINN solution
    # note that we must use tensorflow math primitives such as sin, cos, etc!
    forcing = - (a1 * pi) ** 2 * sin(a1 * pi * x) * sin(a2 * pi * y) - \
              (a2 * pi) ** 2 * sin(a1 * pi * x) * sin(a2 * pi * y) + \
              ksq * sin(a1 * pi * x) * sin(a2 * pi * y)

    f_u = u_xx + u_yy + ksq * u - forcing  # = 0

    return f_u


upper_x = dirichletBC(Domain, val=0.0, var='x', target="upper")
lower_x = dirichletBC(Domain, val=0.0, var='x', target="lower")
upper_y = dirichletBC(Domain, val=0.0, var='y', target="upper")
lower_y = dirichletBC(Domain, val=0.0, var='y', target="lower")

BCs = [upper_x, lower_x, upper_y, lower_y]

layer_sizes = [2, 50, 50, 50, 50, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)

model.fit(tf_iter=10000, newton_iter=10000)

# get exact solution
nx, ny = (1001, 1001)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)

xv, yv = np.meshgrid(x, y)

x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))

# Exact analytical soln is available:
Exact_u = np.sin(math.pi * xv) * np.sin(4 * math.pi * yv)

# Flatten for use
u_star = Exact_u.flatten()[:, None]

# Plotting
x = Domain.domaindict[0]['xlinspace']
y = Domain.domaindict[1]["ylinspace"]

X, Y = np.meshgrid(x, y)

# print(np.shape((X,Y))) # 2, 256, 256
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

lb = np.array([-1.0, -1.0])
ub = np.array([1.0, 1])

u_pred, f_u_pred = model.predict(X_star)

error_u = tdq.helpers.find_L2_error(u_pred, u_star)
print('Error u: %e' % (error_u))

U_pred = tdq.plotting.get_griddata(X_star, u_pred.flatten(), (X, Y))
FU_pred = tdq.plotting.get_griddata(X_star, f_u_pred.flatten(), (X, Y))

lb = np.array([-1.0, -1.0])
ub = np.array([1.0, 1.0])

tdq.plotting.plot_solution_domain1D(model, [x, y], ub=ub, lb=lb, Exact_u=Exact_u)
