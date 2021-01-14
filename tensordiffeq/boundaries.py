from tensordiffeq.domains import *
import numpy as np
import tensorflow as tf
from .utils import multimesh, flatten_and_stack

class BC(DomainND):
    def __init__(self):
        self.doms = self.domain.create_domains()
        self.grid = self.domain.create_mesh()

    def compile(self):
        self.out = self.create_input()

    def predict_values(self, model):
        return model(self.out)

    def loss(self, model):
        preds = predict_values(model)
        return MSE(preds, self.val)


class dirichlectBC(BC):
    def __init__(self, domain, val, var, target):
        self.domain = domain
        self.val = val
        self.var = var
        self.target = target

    def get_dims_list(self):
        linspace_list = []
        iter_ids = np.setdiff1d(self.domain.domain_ids, self.var).tolist()
        for id in (iter_ids):
            self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == id)
            #print(self.domain.domaindict)
            linspace_list.append(self.dict_[(id+"linspace")])
        return linspace_list

    def create_target_input_repeat(self):
        search_key = self.var
        fidelity_key = "fidelity"
        #print(self.domain.domaindict)
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == search_key)
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != search_key]
        fids = []
        for dict_ in self.dicts_:
            res = [val for key, val in dict_.items() if fidelity_key in key]
            fids.append(res)
        reps = np.prod(fids)
        repeated_value = np.repeat(self.dict_[(self.var+self.target)],reps)
        return repeated_value


    def create_input(self):
        repeated_value = self.create_target_input_repeat()
        repeated_value = np.reshape(repeated_value, (-1,1))
        mesh = flatten_and_stack(multimesh(self.get_dims_list()))
        mesh = np.insert(mesh, self.domain.vars.index(self.var), repeated_value.flatten(), axis=1)
        return mesh





class IC(BC):
    def __init__(self, domain, fun, vars):
        self.domain = domain
        self.fun = fun
        self.vars = vars




class periodicBC(BC):
    def __init__(self, domain, var):
        self.domain = domain

    def u_x_model(self, u_model, nn_input):
        u = u_model(nn_input)
        u_x = tf.gradients(u, nn_input[:,0:1])
        return u, u_x

    def create_edges(self):
        edges = []
        for i, val in enumerate(self.domain.bounds[:-1]):
            for value in val:
                edges.append(tf.concat([np.repeat(value, self.domain.fidel[i]), self.doms[-1]], 0))

    def loss(self, u_model):
        loss = 0.0
        for i, val in enumerate(self.val):
            tf.assign_add()
        u_lb_pred, u_x_lb_pred = self.u_x_model(self.u_model, self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.u_model, self.x_ub, self.t_ub)
        return
