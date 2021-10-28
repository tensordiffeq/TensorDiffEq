

import tensordiffeq as tdq
from tensordiffeq.boundaries import *
from tensordiffeq.models import CollocationSolverND
import math
import pytest

def main(args):

    if args is None:
        args = {'layer_sizes': [2, 21, 21, 21, 21, 1],

                'run_functions_eagerly': True,
                'epoch_adam': 20,
                'epoch_lbfgs': 20,
                'lbfgs_eager': False,
                'isAdaptive': True,
                'dist_training': False,
                'dict_adaptive': {"residual": [True],
                                  "BCs": [True, False, False]},
                'N_x': 100,
                'N_t': 50,
                'N_f': 5000,
                'batch_sz': 200,
                }

    layer_sizes = args['layer_sizes']
    run_functions_eagerly = args['run_functions_eagerly']
    epoch_adam = args['epoch_adam']
    epoch_lbfgs = args['epoch_lbfgs']
    lbfgs_eager = args['lbfgs_eager']
    isAdaptive = args['isAdaptive']
    dist_training = args['dist_training']
    dict_adaptive = args['dict_adaptive']
    N_x = args['N_x']
    N_t = args['N_t']
    N_f = args['N_f']
    batch_sz = args['batch_sz']


    tf.config.run_functions_eagerly(run_functions_eagerly)

    Domain = DomainND(["x", "t"], time_var='t')
    Domain.add("x", [-1.0, 1.0], N_x)
    Domain.add("t", [0.0, 1.0], N_t)
    Domain.generate_collocation_points(N_f)

    def func_ic(x):
        return -np.sin(x * math.pi)

    init = IC(Domain, [func_ic], var=[['x']])
    upper_x = dirichletBC(Domain, val=0.0, var='x', target="upper")
    lower_x = dirichletBC(Domain, val=0.0, var='x', target="lower")

    BCs = [init, upper_x, lower_x]

    def f_model(u_model, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u = u_model(tf.concat([x, t], 1))
            u_x = tape.gradient(u, x)

        u_xx = tape.gradient(u_x, x)
        u_t = tape.gradient(u, t)

        f_u = u_t + u * u_x - 0.01 / tf.constant(math.pi) * u_xx

        return f_u

    ## Which loss functions will have adaptive weights
    # "residual" should a tuple for the case of multiple residual equation
    # BCs have to follow the same order as the previously defined BCs list
    dict_adaptive = dict_adaptive

    ## Weights initialization
    # dictionary with keys "residual" and "BCs". Values must be a tuple with dimension
    # equal to the number of residuals and boundary conditions, respectively

    if dict_adaptive["residual"][0] == False:
        init_residual = None
    else:
        init_residual = tf.ones([N_f, 1])

    if dict_adaptive["BCs"][0] == False:
        init_IC = None
    else:
        init_IC = tf.ones([N_x, 1])

    if dict_adaptive["BCs"][1] == False:
        init_BC1 = None
    else:
        init_BC1 = tf.ones([N_t, 1])

    if dict_adaptive["BCs"][2] == False:
        init_BC2 = None
    else:
        init_BC2 = tf.ones([N_t, 1])

    init_weights = {"residual": [init_residual],
                    "BCs": [init_IC, init_BC1, init_BC2]}

    model = CollocationSolverND()
    model.compile(layer_sizes, f_model, Domain, BCs,
                  isAdaptive=isAdaptive,
                  dict_adaptive=dict_adaptive,
                  init_weights=init_weights,
                  dist=dist_training)

    model.fit(tf_iter=epoch_adam,
              newton_iter=epoch_lbfgs,
              newton_eager=lbfgs_eager,
              batch_sz=batch_sz)

    return

if __name__ == "__main__":
    main(args=None)
