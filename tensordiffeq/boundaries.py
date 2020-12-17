import domains
import numpy as np

class BC(Domain):
    def __init__(self, u_model):
        self.u_model = u_model
        self.doms = self.create_domains()
        self.grid = self.create_mesh()

    def predict(self):
        return self.u_model(val)


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

    def u_x_model(self, u_model):
        u = u_model(tf.concat([x,t], 1))
        u_x = tf.gradients(u, x)
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


class IC():
