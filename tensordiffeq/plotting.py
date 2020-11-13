
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
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def plot_solution_domain1D(model, domain, ub, lb, Exact_u = None):

    X, T = np.meshgrid(domain[0],domain[1])

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    if Exact_u is not None:
        u_star = Exact_u.T.flatten()[:,None]

    u_pred, f_u_pred = model.predict(X_star)
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

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$u(t,x)$', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(domain[0],Exact_u[:,len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$' % (domain[1][len_]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(domain[0],Exact_u[:,2*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[2*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = %.2f$' % (domain[1][2*len_]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(domain[0],Exact_u[:,3*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[3*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = %.2f$' % (domain[1][3*len_]), fontsize = 10)

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
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    cbar = plt.colorbar(ec)
    cbar.set_label('$\overline{f}_u$ prediction')
    plt.show()

def get_griddata(grid, data, dims):
    return griddata(grid, data, dims, method='cubic')
