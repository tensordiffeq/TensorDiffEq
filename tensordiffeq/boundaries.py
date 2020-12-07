import domains
import numpy as np

class BC(Domain):
    def __init__(self, u_model):
        self.u_model = u_model

class dirichlectBC(BC):
    def __init__(self, domain, val):
        self.val = val

    def create_domain(self, [vars], ):
        for var in vars:
            np.linspace()
    f

    def predict(self):
        return self.u_model(val)


class periodicBC(BC):
    def __init__(self, u, val):



class IC():
