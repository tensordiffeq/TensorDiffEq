import numpy as np


def creating_cases(flux_types, diffusions, Ms,
                   epoch_adam_std, epoch_lbfgs_std, Adaptive_types,
                   nondaptive_lbfgs=True, dict_adaptive_std=None, init_weights_inside=None, init_weights_outside=None):
    cases = []
    for flux_type in flux_types:
        for M in Ms[flux_type]:
            for Adaptive_type in Adaptive_types:
                # 0 - None (no adaptive method)
                # 1 - Self-adaptive (https://arxiv.org/pdf/2009.04544.pdf),
                # 2 - Self-adaptive_loss with weights for the entire loss function,
                # 3 - NTK (https://arxiv.org/abs/2007.14527)
                if Adaptive_type == 0:  #
                    dict_adaptive = None
                    init_weights = None
                    epoch_adam = epoch_adam_std
                    epoch_lbfgs = epoch_lbfgs_std if nondaptive_lbfgs else 0
                    case = []
                    for diffusion in diffusions[flux_type]:
                        case_temp = {'flux_type': flux_type,
                                     'diffusion': diffusion,
                                     'M': M,
                                     'Adaptive_type': Adaptive_type,
                                     'dict_adaptive': dict_adaptive,
                                     'init_weights': init_weights,
                                     'epoch_adam': epoch_adam,
                                     'epoch_lbfgs': epoch_lbfgs}
                        case.append(case_temp)
                else:
                    dict_adaptive = dict_adaptive_std
                    if Adaptive_type == 1:  #
                        init_weights = init_weights_inside
                    elif Adaptive_type == 2:  #
                        init_weights = init_weights_outside

                    case = {'flux_type': flux_type,
                            'diffusion': None,
                            'M': M,
                            'Adaptive_type': Adaptive_type,
                            'dict_adaptive': dict_adaptive,
                            'init_weights': init_weights,
                            'epoch_adam': epoch_adam_std,
                            'epoch_lbfgs': epoch_lbfgs_std}

                if isinstance(case, list):
                    for case_in in case:
                        cases.append(case_in)
                else:
                    cases.append(case)

    return cases


def creating_x_star(Domain):
    x = Domain.domaindict[0]['xlinspace']
    t = Domain.domaindict[1]["tlinspace"]

    # create mesh for plotting
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    return X_star
