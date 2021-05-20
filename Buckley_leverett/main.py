# first implementation of TensorDiffeq
# https://sciml.tamids.tamu.edu/scientific-machine-learning-lab/software/

import os
import sys
import pickle
import random
from datetime import datetime
import time

import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import *

from Analytical import u_analytical
from main_printing import *


if sys.gettrace() is not None:
    # https://intellij-support.jetbrains.com/hc/en-us/community/posts/205819799-Way-for-my-Python-code-to-detect-if-it-s-being-run-in-the-debugger-
    # Hi, you could check the result of sys.gettrace() is None.
    # That will mean that there is no PyCharm debugger involved,
    # not another one.
    tf.config.run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'

# type of relative permeabilities

# ***********************************************************************
#                    1 - Generating a Domain
# ***********************************************************************

# Instantiate
Domain = DomainND(["x", "t"], time_var='t')

# Adding variables to your domain
# add(token, vals, fidel)
#   - fidel - An int defining the level of fidelity of the evenly spaced
#   samples along this dimensions boundary points
Domain.add("x", [0.0, 1.0], 512)
Domain.add("t", [0.0, 1.0], 201)
# Domain.add("x", [0.0, 1.0], 250)
# Domain.add("t", [0.0, 1.0], 150)
# Generation of Collocation Points
#   N_f is an int describing the number of collocation points desired
#   within the domain defined in your DomainND object
N_f = 10000
# N_f = 250
Domain.generate_collocation_points(N_f)


# ***********************************************************************
#                    2 - Describe the physics of the model
# ***********************************************************************
def f_model_complete(flux_type, diffusion, M):
    def residual(u_model, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)

            u = u_model(tf.concat([x, t], 1))

            if flux_type == 'concave':
                f = u / (u + (1. - u) / M)
            elif flux_type == 'non-convex':
                f = (u ** 2.) / (u ** 2 + (1. - u) ** 2. / M)
            else:  # convex
                f = u ** 2.

            if diffusion is not None:
                u_x = tape.gradient(u, x)

        u_t = tape.gradient(u, t)
        f_x = tape.gradient(f, x)
        r = u_t + f_x

        if diffusion is not None:
            u_xx = tape.gradient(u_x, x)
            r = r - diffusion * u_xx
        return r

    return residual


# ***********************************************************************
#                    3 - Initial Conditions
# ***********************************************************************
def func_ic_x(x):
    return tdq.utils.constant(0)


init = IC(Domain, [func_ic_x], var=[['x']])

# ***********************************************************************
#                    3 - Boundary Conditions
# ***********************************************************************
lower_x = dirichletBC(Domain, val=1.0, var='x', target="lower")

BCs = [init, lower_x]
# ***********************************************************************
#                    4 - Creating the ANN
# ***********************************************************************
layer_sizes = [2, 500, 1]

# ***********************************************************************
#                    5 - Config to cases to test
# ***********************************************************************
#  Relative permeabilities type
flux_types = ['concave', 'non-convex', 'convex']

## Diffusion term
diffusion = {'concave': [None],
             'non-convex': [None, 1.0e-2],  # [None, 1.0e-2, 2.5e-3, 1.0e-3],
             'convex': [None, 1.0e-2]}
## ratio of phase viscosity: mu_o/mu_w
M = {'concave': 2,
     'non-convex': 1,
     'convex': None}

# ***********************************************************************
#                    6 - Important definitions
# **********************************************************************
hprc_jobID = '_51142'
flag_fitting = False
flag_saving = False
flag_plotting = True
epoch_adam_std = 10000
epoch_lbfgs_std = 10000


x = Domain.domaindict[0]['xlinspace']
t = Domain.domaindict[1]["tlinspace"]

# create mesh for plotting
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# ***********************************************************************
#                    7 - Fitting
# ***********************************************************************
predictions = {key: {} for key in flux_types}
losses = {key: {} for key in flux_types}

if flag_fitting:
    print("Starting optimization at ", datetime.now().strftime("%H:%M:%S"))
    tic = time.perf_counter()

    for flux_type in flux_types:
        n_dif = len(diffusion[flux_type])
        for i in range(n_dif + 1):
            if i == 0:
                epsilon = None
                epoch_adam = epoch_adam_std
                epoch_lbfgs = epoch_lbfgs_std
                Adaptive_types = [3]

            else:
                epsilon = diffusion[flux_type][i - 1]
                col_weights = None
                u_weights = None
                epoch_adam = epoch_adam_std
                epoch_lbfgs = 0
                Adaptive_types = [0]

                outside_test = [False]

            f_model = f_model_complete(flux_type, epsilon, M[flux_type])

            for Adaptive_type in Adaptive_types:

                if Adaptive_type == 0:
                    dict_adaptive = None
                    init_weights = None
                else:
                    dict_adaptive = {"residual": [True],
                                     "BCs": [True, True]}

                    if Adaptive_type == 2 or Adaptive_type == 3:
                        init_weights = {"residual": [tf.ones([1, 1])],
                                        "BCs": [tf.ones([1, 1]), tf.ones([1, 1])]}
                    elif Adaptive_type == 1:
                        init_weights = {"residual": [tf.ones([N_f, 1])],
                                        "BCs": [tf.ones([512, 1]), tf.ones([201, 1])]}


                model = CollocationSolverND(verbose=False)
                model.compile(layer_sizes, f_model, Domain, BCs, Adaptive_type=Adaptive_type,
                              dict_adaptive=dict_adaptive, init_weights=init_weights)

                # Train the model
                model.fit(tf_iter=epoch_adam, newton_iter=epoch_lbfgs)

                ## Forward pass through model
                u_pred, f_u_pred = model.predict(X_star)


                # Saving outputs for printing
                solver_method: str = 'Adaptive' if Adaptive_type != 0 else f'e_{epsilon}'
                adaptive_test: str = 'Outside' if Adaptive_type == 2 else 'inside'

                solver_method: str = solver_method if Adaptive_type == 0 else f'{solver_method}_{adaptive_test}'

                predictions[flux_type].update({solver_method: (u_pred, f_u_pred)})

                losses[flux_type].update({solver_method: model.losses})

                #del model

    print("Finishing optimization at ", datetime.now().strftime("%H:%M:%S"))
    toc = time.perf_counter()
    second = 1.0
    minute = 60.0 * second
    hour = 60 * minute
    day = 24 * hour
    print(f"Elapsed time of  {(toc - tic) / minute:0.4f} minutes")

if flag_saving:
    save_results = (predictions, losses)
    # Save pickle with all predictions

    with open(fr'Results/output{hprc_jobID}.pickle', 'wb') as f:
        pickle.dump(save_results, f)

# ***********************************************************************
#                    8 - Plotting
# ***********************************************************************
if flag_plotting:

    if flag_fitting is False:


        with open(fr'Results/output{hprc_jobID}.pickle', 'rb') as f:
            predictions, losses = pickle.load(f)

    n_cases = len(predictions)

    Exact_u = {}
    for case, preds in predictions.items():
        # Analytical solutions
        Exact_u[case] = u_analytical(case, M[case], X.flatten(), T.flatten())

        subcases = list(preds.keys())
        preds_list = list(preds.values())

        u_pred = np.hstack(list(zip(*preds_list))[0])
        f_u_pred = np.hstack(list(zip(*preds_list))[1])

        # if Exact_u is not None:
        #     u_star = Exact_u.T.flatten()[:, None]
        #     error_u = tdq.helpers.find_L2_error(u_pred, u_star)
        #     print('Error u: %e' % (error_u))

        lb = np.array([0.0, 0.0])  # x_inf, t_inf
        ub = np.array([1.0, 1.0])  # x_sup, t_sup

        ## Plot losses
        # tf.config.run_functions_eagerly(True)
        # plot_losses(losses[case], title=case.capitalize()+hprc_jobID, divider=epoch_adam_std,
        #                          xlim=epoch_adam_std + epoch_lbfgs_std)

        ## Plot predictions
        plot_solution_domain1D_v2([u_pred, f_u_pred], [x, t],
                                               ub=ub, lb=lb, Title=case+hprc_jobID, Legends=subcases,
                                               Exact_u=Exact_u[case])

## Make a sound when finish
# import winsound

# duration = 1000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)
