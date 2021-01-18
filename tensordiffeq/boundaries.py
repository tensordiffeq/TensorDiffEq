from tensordiffeq.domains import *
import numpy as np
import tensorflow as tf
from .utils import multimesh, flatten_and_stack, MSE


def get_linspace(dict_):
    lin_key = "linspace"
    return [val for key, val in dict_.items() if lin_key in key][0]


class BC(DomainND):
    def __init__(self):
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.var)

    def compile(self):
        self.input = self.create_input()

    def predict_values(self, model):
        self.preds = model(self.input)

    def loss(self):
        return MSE(self.preds, self.val)

    def get_dict(self, var):
        return next(item for item in self.domain.domaindict if item["identifier"] == var)

    def get_not_dims(self, var):
        dims = []
        for dict_ in self.dicts_:
            dims.append(get_linspace(dict_))
        return dims


class dirichlectBC(BC):
    def __init__(self, domain, val, var, target):
        self.domain = domain
        self.val = val
        self.var = var
        self.target = target
        super().__init__()
        self.compile()

    def create_target_input_repeat(self):
        fidelity_key = "fidelity"
        # print(self.domain.domaindict)
        fids = []
        for dict_ in self.dicts_:
            res = [val for key, val in dict_.items() if fidelity_key in key]
            fids.append(res)
        reps = np.prod(fids)
        repeated_value = np.repeat(self.dict_[(self.var + self.target)], reps)
        return repeated_value

    def create_input(self):
        repeated_value = self.create_target_input_repeat()
        repeated_value = np.reshape(repeated_value, (-1, 1))
        mesh = flatten_and_stack(multimesh(self.get_not_dims(self.var)))
        mesh = np.insert(mesh, self.domain.vars.index(self.var), repeated_value.flatten(), axis=1)
        return mesh


def get_function_out(func, var, dict_):
    linspace = get_linspace(dict_)
    return func(linspace)


class IC(BC):
    def __init__(self, domain, fun, var):
        self.domain = domain
        self.fun = fun
        self.vars = var
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.domain.time_var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.domain.time_var)
        self.compile()
        # super().__init__()

    def create_input(self):
        dims = self.get_not_dims(self.domain.time_var)
        # vals = np.reshape(fun_vals, (-1, len(self.vars)))
        mesh = flatten_and_stack(multimesh(dims))
        t_repeat = np.repeat(0.0, len(mesh))
        mesh = np.concatenate((mesh, np.reshape(t_repeat, (-1, 1))), axis=1)
        print(mesh)
        return mesh

    def create_target(self):
        fun_vals = []
        for i, var_ in enumerate(self.vars):
            arg_list = []
            for j, var in enumerate(var_):
                var_dict = self.get_dict(var)
                arg_list.append(get_linspace(var_dict))
            inp = flatten_and_stack(multimesh(arg_list))
            fun_vals.append(self.fun[i](*inp.T))


class periodicBC(BC):
    def __init__(self, domain, var):

        self.domain = domain

    def u_x_model(self, u_model, nn_input):
        u = u_model(nn_input)
        u_x = tf.gradients(u, nn_input[:, 0:1])
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
