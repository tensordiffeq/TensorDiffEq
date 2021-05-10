# Raissi et al plotting scripts - https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py
# All code in this script is credited to Raissi et al


import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


import matplotlib.pyplot as plt

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def plot_solution_domain1D(model, domain, ub, lb, Exact_u=None, u_transpose=False):
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
    X, T = np.meshgrid(domain[0],domain[1])

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    if Exact_u is not None:
        u_star = Exact_u.T.flatten()[:,None]

    u_pred, f_u_pred = model.predict(X_star)
    if u_transpose:
        U_pred = griddata(X_star, u_pred.T.flatten(), (X, T), method='cubic')
    else:
        U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    fig, ax = newfig(1.3, 1.0)

    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    len_ = len(domain[1])//4

    line = np.linspace(domain[0].min(), domain[0].max(), 2)[:,None]
    ax.plot(domain[1][len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][2*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][3*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('u(t,x)', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(domain[0],Exact_u[:,len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title('t = %.2f' % (domain[1][len_]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(domain[0],Exact_u[:,2*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[2*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('t = %.2f' % (domain[1][2*len_]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(domain[0],Exact_u[:,3*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[3*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('t = %.2f' % (domain[1][3*len_]), fontsize = 10)

    plt.show()


def plot_weights(model, scale = 1):
    plt.scatter(model.t_f, model.x_f, c = model.col_weights.numpy(), s = model.col_weights.numpy()/float(scale))
    plt.show()

def plot_glam_values(model, scale = 1):
    plt.scatter(model.t_f, model.x_f, c = model.g(model.col_weights).numpy(), s = model.g(model.col_weights).numpy()/float(scale))
    plt.show()

def plot_residuals(FU_pred, extent):
    fig, ax = plt.subplots()
    ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
                extent=extent,
                origin='lower', aspect='auto')

    #ax.add_collection(ec)
    ax.autoscale_view()
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    cbar = plt.colorbar(ec)
    cbar.set_label('\overline{f}_u prediction')
    plt.show()

def get_griddata(grid, data, dims):
    return griddata(grid, data, dims, method='cubic')

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
    figsize = (width, 4.5)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Creading figure
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(wspace=wspace)

    for i, ax in enumerate(axs.flatten(), 1):
        if Exact_u is not None:
            exact_u = Exact_u[Title]
            if ~np.all(exact_u == 0.0):
                exact_u = np.reshape(exact_u, T.shape).T
                x_domain = domain[0]
                u_analytical = exact_u[:, i * len_]
                idx_zero = x_domain == 0.0
                x_domain = x_domain[~idx_zero]
                u_analytical = u_analytical[~idx_zero]
                ax.plot(x_domain, u_analytical, linestyle='-', linewidth=2, label='Analytical')

        for j, u_pred in enumerate(u_preds.T):
            if u_transpose:
                U_pred = griddata(X_star, u_pred.T.flatten(), (X, T), method='cubic')
            else:
                U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

            if Legends[j] == 'Adaptive':
                linestyle = ':'
                marker = 'x'
            else:
                linestyle = '--'
                marker = None
            ax.plot(x, U_pred[i * len_, :], linestyle=linestyle, marker=marker, linewidth=2, label=Legends[j])

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
