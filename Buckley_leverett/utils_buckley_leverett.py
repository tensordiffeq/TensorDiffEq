import numpy as np
import matplotlib.pyplot as plt
from tensordiffeq.helpers import find_L2_error


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
                            'diffusion': diffusion,
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


def plot_solution_domain1D_v2(preds, domain, ub, lb, Exact_u=None, u_transpose=False, Title=None, Legends=None):
    """
    Plot a 1D solution Domain
    Arguments
    ---------
    model : model
        a `model` class which contains the PDE solution
    domain : Domain
        a `Domain` object containing the x,t pairs
    ub: list
        a list of floats containing the upper boundaries of the plot
    lb : list
        a list of floats containing the lower boundaries of the plot
    Exact_u : list
        a list of the exact values of the solution for comparison
    u_transpose : Boolean
        a `bool` describing whether or not to transpose the solution plot of the domain
    Returns
    -------
    None
    """
    x, t = domain
    X, T = np.meshgrid(x, t)
    u_preds, f_u_preds = preds

    len_ = t.shape[0] // 4

    width = 0
    width_ratios = []
    for i in range(3):
        width_ratios.append(15)
        width += 2.5
    width_ratios.append(1)
    wspace = 0.0
    figsize = (width, 4.5 * 2)
    figsize = (8, 3)

    # Creading figure
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(wspace=wspace)

    for i, ax in enumerate(axs.flatten(), 1):
        textstr = 'L2 Error\n'
        if Exact_u is not None:
            exact_u = Exact_u
            if ~np.all(exact_u == 0.0):
                exact_u = np.reshape(exact_u, T.shape).T
                x_domain = domain[0]
                u_analytical = exact_u[:, i * len_]
                idx_zero = x_domain == 0.0
                x_domain = x_domain[~idx_zero]
                u_analytical = u_analytical[~idx_zero]
                ax.plot(x_domain, u_analytical, linestyle='-', linewidth=2,
                        label='Analytical', zorder=0, color='blue')

        for j, u_pred in enumerate(u_preds.T):
            n_x = x.shape[0]
            n_t = t.shape[0]
            U_pred = np.reshape(u_pred, (n_t, n_x), order='C')

            if Legends[j][0:3] == 'Sel':
                linestyle = ':'
                marker = None
                zorder = 10
                color = 'green'
            else:
                linestyle = '--'
                marker = None
                zorder = 1
                color = 'orange'

            legend_j = Legends[j]

            if legend_j != 'Adaptive_Lagrangian':
                ax.plot(x, U_pred[i * len_, :], linestyle=linestyle, marker=marker, linewidth=2, label=legend_j,
                        zorder=zorder, color=color)

            if Exact_u is not None:
                # u_star = Exact_u.T.flatten()[:, None]
                # error_u = find_L2_error(u_pred, exact_u[:, i * len_])
                error_u = find_L2_error(U_pred[i * len_, :], exact_u[:, i * len_])

                if legend_j.lower()[0:3] == 'sel':
                    l2_title = 'sa'
                else:
                    l2_title = legend_j.lower()
                textstr = textstr + f'{l2_title}: {error_u:.2e}\n'

        if textstr is not None:
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title('t = %.2f' % (t[i * len_]), fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('saturation(t,x)')
        # ax.axis('square')
        ax.set_xlim([-lb[0] - .1, ub[0] + .1])
        ax.set_ylim([-lb[1] - .1, ub[1] + .1])

    ## L2 error total
    error_total = ''
    for j, u_pred in enumerate(u_preds.T):
        error_u = find_L2_error(u_pred, Exact_u)
        error_total = error_total + f'\nL2: {Legends[j].lower()} : {error_u:.3e}'


    ax.legend(fontsize=8, loc='best', framealpha=0.5)
    fig.suptitle(Title + error_total)
    plt.tight_layout()
    plt.savefig(fr'../figs/{Title}.png')
    plt.show()


def plot_losses(losses_main, title=None, divider=None, xlim=None, ylim=None):
    # TODO take a look on pip install tensorflow-plot

    if isinstance(losses_main, dict):
        n_losses = len(losses_main)
    else:
        n_losses = 1

    fig_height = n_losses * 4.8
    figsize = (8, fig_height)

    fig, axes = plt.subplots(nrows=n_losses, ncols=1, figsize=(4, 6))

    for i, (subcase, losses) in enumerate(losses_main.items()):

        if n_losses == 1:
            ax = axes
        else:
            ax = axes[i]

        n_epochs = len(losses)
        n_loss = len(losses[0].keys())
        epochs = np.arange(n_epochs)

        for i, loss_type in enumerate(losses[0].keys()):
            loss = [np.asarray(loss_epoch[loss_type]).flatten() for loss_epoch in losses]
            if i == n_loss - 1:
                linestyle = 'dotted'
            else:
                linestyle = "solid"

            # ax.plot(epochs, loss, label=loss_type, linewidth=3, linestyle=linestyle)
            ax.plot(epochs, loss, label=loss_type, linewidth=2, linestyle=linestyle)

        if divider is not None:
            ax.axvline(x=divider, linestyle='dashed', color='black')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_xlim(right=xlim)
        # ax.set_ylim(ylim)
        ax.set_title(subcase, fontsize=12)
        ax.set_yscale('log')
        ax.grid()
    # ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.5)
    ax.legend(fontsize=8, loc='best', framealpha=0.5)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(fr'../figs/losses_{title}.png')
    plt.show()
