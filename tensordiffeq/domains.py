import numpy as np
from .utils import LatinHypercubeSample

class Rectangle1D:
    def __init__(self, xlim, tlim=None):
        self.x_ub = xlim[0]
        self.x_lb = xlim[1]
        if tlim is not None:
            self.t0 = tlim[0]
            self.tmax = tlim[1]


class Rectangle2D(Rectangle1D):
    def __init__(self, xlim, ylim, tlim=None):
        super().__init__(self, xlim, tlim=tlim)
        self.y_ub = ylim[0]
        self.y_lb = ylim[1]


class Rectangle3D(Rectangle2D):
    def __init__(self, xlim, ylim, zlim, tlim=None):
        super().__init__(self, xlim, ylim, tlim=tlim)
        self.z_ub = zlim[0]
        self.z_lb = zlim[1]


class DomainND:
    def __init__(self, var, time_var=None):
        self.vars = var
        self.domaindict = []
        self.domain_ids = []
        self.time_var = time_var

    # def create_domains(self):
    #     doms = []
    #     for i, val in self.vals:
    #         doms.append(np.linspace(val[0], val[1], self.fidel[i]))
    #     return doms
    #
    # def create_mesh(self, doms):
    #     mesh = np.meshgrid(doms)
    #     return mesh

    def generate_collocation_points(self, N_f):
        range_list = []
        for dict_ in self.domaindict:
            range_list.append([val for key, val in dict_.items() if "range" in key][0])
        limits = np.array(range_list)  # x,t domain
        X_f = LatinHypercubeSample(N_f, limits)
        self.X_f = X_f

    def add(self, token, vals, fidel):
        self.domain_ids.append(token)
        self.domaindict.append({
            "identifier": token,
            "range": vals,
            (token + "fidelity"): fidel,
            (token + "linspace"): np.linspace(vals[0], vals[1], fidel),
            (token + "upper"): vals[1],
            (token + "lower"): vals[0]
        })
