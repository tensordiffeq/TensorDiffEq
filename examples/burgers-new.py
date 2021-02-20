import math
import scipy.io
import tensordiffeq as tdq
from tensordiffeq.boundaries import *
from tensordiffeq.models import CollocationSolverND

Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 256)
Domain.add("t", [0.0, 1.0], 100)

N_f = 10000
Domain.generate_collocation_points(N_f)


def func_ic(x):
    return -np.sin(x * math.pi)

init = IC(Domain, [func_ic], var=[['x']])
upper_x = dirichletBC(Domain, val=0.0, var='x', target="upper")
lower_x = dirichletBC(Domain, val=0.0, var='x', target="lower")

BCs = [init, upper_x, lower_x]


def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    f_u = u_t + u * u_x - (0.01 / tf.constant(math.pi)) * u_xx
    return f_u


layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)

# to reproduce results from Raissi and the SA-PINNs paper, train for 10k newton and 10k adam
model.fit(tf_iter=10000, newton_iter=10000)


#######################################################
#################### PLOTTING #########################
#######################################################

data = scipy.io.loadmat('burgers_shock.mat')

Exact = data['usol']
Exact_u = np.real(Exact)

# t = data['tt'].flatten()[:,None]
# x = data['x'].flatten()[:,None]

x = Domain.domaindict[0]['xlinspace']
t = Domain.domaindict[1]["tlinspace"]

X, T = np.meshgrid(x, t)

# print(np.shape((X,T))) #2, 100, 256
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

u_pred, f_u_pred = model.predict(X_star)

error_u = tdq.helpers.find_L2_error(u_pred, u_star)
print('Error u: %e' % (error_u))

U_pred = tdq.plotting.get_griddata(X_star, u_pred.flatten(), (X, T))
FU_pred = tdq.plotting.get_griddata(X_star, f_u_pred.flatten(), (X, T))

lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

tdq.plotting.plot_solution_domain1D(model, [x, t], ub=ub, lb=lb, Exact_u=Exact_u)
