import domains
import numpy as np

class BC(Domain):
    def __init__(self, u_model):
        self.u_model = u_model

    def create_domain(self, domain, fidel):
        return np.linspace(domain[0], domain[1], fidel)

    def predict(self):
        return self.u_model(val)

class dirichlectBC(BC):
    def __init__(self, domain, val):
        self.val = val



class IC(BC):
    def __init__(self, domain, fun, vars):
        self.domain = domain
        self.fun = fun
        self.vars = vars




class periodicBC(BC):
    def __init__(self, u, val):



class IC():
