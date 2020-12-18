import domains
import numpy as np

class BC(Domain):
    def __init__(self):
        self.doms = self.domain.create_domains()
        self.grid = self.domain.create_mesh()


class dirichlectBC(BC):
    def __init__(self, domain, val):
        self.domain = domain
        self.val = val


class IC(BC):
    def __init__(self, domain, fun, vars):
        self.domain = domain
        self.fun = fun
        self.vars = vars




class periodicBC(BC):
    def __init__(self, domain):
        self.domain = domain

    def u_x_model(self, u_model, nn_input):
        u = u_model(nn_input)
        u_x = tf.gradients(u, nn_input[:,0:1])
        return u, u_x

    def create_edges(self):
        edges = []
        for i, val in enumerate(self.domain.bounds[:-1]):
            for value in val:
                edges.append(tf.concat([np.repeat(value, self.domain.fidel[i]), self.doms[-1]], 1))

    def loss(self, u_model):
        loss = 0.0
        for i, val in enumerate(self.val):
            tf.assign_add()
        u_lb_pred, u_x_lb_pred = self.u_x_model(self.u_model, self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.u_model, self.x_ub, self.t_ub)
        return
