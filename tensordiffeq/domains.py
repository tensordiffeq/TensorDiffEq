import numpy as np
from .utils import LatinHypercubeSample


class DomainND:
    def __init__(self, var, time_var=None):
        self.vars = var
        self.domaindict = []
        self.domain_ids = []
        self.time_var = time_var

    def generate_collocation_points(self, N_f):
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]

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
