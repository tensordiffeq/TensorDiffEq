from tensordiffeq.domains import DomainND
import numpy as np
import tensorflow as tf
from .utils import multimesh, flatten_and_stack, MSE, convertTensor, get_tf_model


def get_linspace(dict_):
    lin_key = "linspace"
    return [val for key, val in dict_.items() if lin_key in key][0]


class BC(DomainND):
    def __init__(self):
        self.isPeriodic = False
        self.isInit = False


    def compile(self):
        self.input = self.create_input()

    def preds_init(self, model):
        self.preds = model(self.input)

    @tf.function
    def update_values(self, model):
        self.preds = model(self.input)

    def get_dict(self, var):
        return next(item for item in self.domain.domaindict if item["identifier"] == var)

    def get_not_dims(self, var):
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != var]
        return [get_linspace(dict_) for dict_ in self.dicts_]

    def create_target_input_repeat(self, var, target):
        fidelity_key = "fidelity"
        fids = []
        for dict_ in self.dicts_:
            res = [val for key, val in dict_.items() if fidelity_key in key]
            fids.append(res)
        reps = np.prod(fids)
        if target is str:
            return np.repeat(self.dict_[(var + target)], reps)
        else:
            return np.repeat(target, reps)


class dirichletBC(BC):
    def __init__(self, domain, val, var, target):
        self.domain = domain
        self.val = val
        self.var = var
        self.target = target
        super().__init__()
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.var)
        self.target = self.dict_[var+target]
        self.compile()

    def create_input(self):
        repeated_value = self.create_target_input_repeat(self.var, self.target)
        repeated_value = np.reshape(repeated_value, (-1, 1))
        mesh = flatten_and_stack(multimesh(self.get_not_dims(self.var)))
        mesh = np.insert(mesh, self.domain.vars.index(self.var), repeated_value.flatten(), axis=1)
        return mesh

    def loss(self):
        return MSE(self.preds, self.val)


def get_function_out(func, var, dict_):
    linspace = get_linspace(dict_)
    return func(linspace)


class IC(BC):
    def __init__(self, domain, fun, var):
        self.isPeriodic = False
        self.isInit = True
        self.domain = domain
        self.fun = fun
        self.vars = var
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.domain.time_var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.domain.time_var)
        self.compile()
        self.create_target()

    def create_input(self):
        dims = self.get_not_dims(self.domain.time_var)
        # vals = np.reshape(fun_vals, (-1, len(self.vars)))
        mesh = flatten_and_stack(multimesh(dims))
        t_repeat = np.repeat(0.0, len(mesh))
        mesh = np.concatenate((mesh, np.reshape(t_repeat, (-1, 1))), axis=1)
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
        self.val = convertTensor(np.reshape(fun_vals, (-1, 1)))

    def loss(self):
        return MSE(self.preds, self.val)

class periodicBC(BC):
    def __init__(self, domain, var, deriv_model):
        self.domain = domain
        self.var = var
        super().__init__()

        self.deriv_model = [get_tf_model(model) for model in deriv_model]
        self.isPeriodic = True
        #self.dicts_ = [item for item in self.domain.domaindict]
        #self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == var)
        self.compile()

    def get_input_upper_lower(self, var):
        #for var in self.dict_["range"]:
        self.upper_repeat = self.create_target_input_repeat(var, self.dict_["range"][1])
        self.lower_repeat = self.create_target_input_repeat(var, self.dict_["range"][0])

    def compile(self):
        self.upper = []
        self.lower = []
        for var in self.var:
            self.dicts_ = [item for item in self.domain.domaindict if item["identifier"] != var]
            self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == var)
            self.get_input_upper_lower(var)
            mesh = flatten_and_stack(multimesh(self.get_not_dims(var)))
            self.upper.append(np.insert(mesh, self.domain.vars.index(var), self.upper_repeat.flatten(), axis=1))
            self.lower.append(np.insert(mesh, self.domain.vars.index(var), self.lower_repeat.flatten(), axis=1))
        outer = []
        for i, lst in enumerate(self.upper):
            tmp = [convertTensor(np.reshape(vec, (-1,1))) for vec in lst.T]
            outer.append(np.asarray(tmp))
        self.upper = outer

        outer = []
        for i, lst in enumerate(self.lower):
            tmp = [np.reshape(vec, (-1,1)) for vec in lst.T]
            outer.append(np.asarray(tmp))
        self.lower = outer

    def u_x_model(self, u_model, inputs):
        return [model(u_model, *inputs) for model in self.deriv_model]


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

    # TODO Add Neumann BC



