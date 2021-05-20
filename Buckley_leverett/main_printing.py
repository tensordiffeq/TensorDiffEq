import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# test

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
    figsize = (width, 4.5*2)
    figsize = (8, 3)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Creading figure
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(wspace=wspace)

    for i, ax in enumerate(axs.flatten(), 1):
        if Exact_u is not None:
            exact_u = Exact_u
            if ~np.all(exact_u == 0.0):
                exact_u = np.reshape(exact_u, T.shape).T
                x_domain = domain[0]
                u_analytical = exact_u[:, i * len_]
                idx_zero = x_domain == 0.0
                x_domain = x_domain[~idx_zero]
                u_analytical = u_analytical[~idx_zero]
                ax.plot(x_domain, u_analytical, linestyle='-', linewidth=2, label='Analytical', zorder = 0)

        for j, u_pred in enumerate(u_preds.T):
            if u_transpose:
                U_pred = griddata(X_star, u_pred.T.flatten(), (X, T), method='cubic')
            else:
                U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')


            if Legends[j][0:3] == 'Ada':
                linestyle = ':'
                marker = None
                zorder = 10
            else:
                linestyle = '--'
                marker = None
                zorder = 1

            legend_j =Legends[j]

            if legend_j == 'Adaptive_inside':
                legend_j = 'Self-adaptive'
            elif legend_j == 'e_None':
                legend_j = 'PINN Baseline'

            if legend_j != 'Adaptive_Outside':
                ax.plot(x, U_pred[i * len_, :], linestyle=linestyle, marker=marker, linewidth=2, label=legend_j,zorder=zorder)

        ax.set_title('t = %.2f' % (t[i * len_]), fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('saturation(t,x)')
        # ax.axis('square')
        ax.set_xlim([-lb[0] - .1, ub[0] + .1])
        ax.set_ylim([-lb[1] - .1, ub[1] + .1])
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)
    ax.legend(loc='best')
    fig.suptitle(Title)
    plt.tight_layout()
    plt.savefig(fr'figs/{Title}.png')
    plt.show()

def plot_losses(losses_main, title=None, divider=None, xlim=None, ylim = None):
    # TODO take a look on pip install tensorflow-plot

    if isinstance(losses_main, dict):
        n_losses = len(losses_main)
    else:
        n_losses = 1

    fig_height = n_losses*4.8
    figsize = (6.4, fig_height)

    fig, axes = plt.subplots(nrows=n_losses, ncols=1, figsize=figsize )

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
        #ax.set_ylim(ylim)
        ax.set_title(subcase, fontsize=12)
        ax.set_yscale('log')
        ax.grid()
    ax.legend(fontsize=10, loc='best')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(fr'figs/losses_{title}.png')
    plt.show()
