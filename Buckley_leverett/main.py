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

from utils_buckley_leverett import *

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

# Generation of Collocation Points
#   N_f is an int describing the number of collocation points desired
#   within the domain defined in your DomainND object
N_f = 10000

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
layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# ***********************************************************************
#                    5 - Important definitions
# **********************************************************************
flag_saving = True
epoch_adam_std = 10000
epoch_lbfgs_std = 10000

# ***********************************************************************
#                    6 - Config to cases to test
# ***********************************************************************
#  Relative permeabilities type
flux_types = ['concave', 'non-convex', 'convex']
flux_types = ['non-convex']

## Diffusion term
diffusions = {'concave': [None],
              'non-convex': [2.5e-3],  # [None, 1.0e-2, 2.5e-3, 1.0e-3],
              'convex': [None]}  # [None, 2.5e-3],

## ratio of phase viscosity: mu_o/mu_w
Ms = {'concave': [2],
      'non-convex': [1],
      'convex': [None]}

## Diffusion term
Adaptive_types = [0, 1]

dict_adaptive = {"residual": [True],
                 "BCs": [True, False]}

init_weights_inside = {"residual": [tf.ones([N_f, 1])],
                       "BCs": [tf.ones([512, 1]), tf.ones([201, 1])]}

init_weights_outside = {"residual": [tf.ones([1, 1])],
                        "BCs": [tf.ones([1, 1]), tf.ones([1, 1])]}

# creating test dictionary
case_tests = creating_cases(flux_types, diffusions, Ms,
                            epoch_adam_std, epoch_lbfgs_std, Adaptive_types,
                            nondaptive_lbfgs=False, dict_adaptive_std=dict_adaptive,
                            init_weights_inside=init_weights_inside, init_weights_outside=init_weights_outside)

# ***********************************************************************
#                    7 - Fitting
# ***********************************************************************

X_star = creating_x_star(Domain)

predictions = {key: {} for key in flux_types}
losses = {key: {} for key in flux_types}

print(f'Number of cases to test: {len(case_tests)}')

print("Starting optimization at ", datetime.now().strftime("%H:%M:%S"))
tic = time.perf_counter()

for counter, case in enumerate(case_tests, 1):
    print(f'Case {counter} of {len(case_tests)}')

    flux_type = case['flux_type']
    diffusion = case['diffusion']
    M = case['M']
    Adaptive_type = case['Adaptive_type']
    dict_adaptive = case['dict_adaptive']
    init_weights = case['init_weights']
    epoch_adam = case['epoch_adam']
    epoch_lbfgs = case['epoch_lbfgs']

    # Physical Model
    f_model = f_model_complete(flux_type, diffusion, M)

    # Create an instance
    model = CollocationSolverND(verbose=False)
    model.compile(layer_sizes, f_model, Domain, BCs, Adaptive_type=Adaptive_type,
                  dict_adaptive=dict_adaptive, init_weights=init_weights)

    # Train the model
    model.fit(tf_iter=epoch_adam, newton_iter=epoch_lbfgs)

    # Forward pass through model
    u_pred, f_u_pred = model.predict(X_star)

    # Saving outputs for printing
    solver_method: str = 'Adaptive' if Adaptive_type != 0 else f'e_{diffusion}'
    adaptive_test: str = 'Self_adaptive' if Adaptive_type == 1 else 'Lagrangian'

    solver_method: str = solver_method if Adaptive_type == 0 else f'{adaptive_test}'

    predictions[flux_type].update({solver_method: (u_pred, f_u_pred)})
    losses[flux_type].update({solver_method: model.losses})

    if flag_saving:
        x = Domain.domaindict[0]['xlinspace']
        t = Domain.domaindict[1]["tlinspace"]


        save_results = (predictions, losses, Ms, [x , t], case_tests)
        # Save pickle with all predictions
        with open(fr'../Results/output.pickle', 'wb') as f:
            pickle.dump(save_results, f)

    del model

print("Finishing optimization at ", datetime.now().strftime("%H:%M:%S"))
toc = time.perf_counter()
print(f"Elapsed time of  {(toc - tic) / 60:0.4f} minutes")


