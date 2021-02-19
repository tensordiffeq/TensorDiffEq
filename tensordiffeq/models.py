import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *
from .fit import *


class CollocationSolverND:
    def __init__(self, assimilate=False):
        self.assimilate = assimilate

    def compile(self, layer_sizes, f_model, domain, bcs, isAdaptive=False,
                col_weights=None, u_weights=None, g=None, dist=False):
        self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.layer_sizes = layer_sizes
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.bcs = bcs
        self.f_model = get_tf_model(f_model)
        self.isAdaptive = False
        self.g = g
        self.domain = domain
        self.dist = dist
        self.col_weights = col_weights
        self.u_weights = u_weights
        self.X_f_dims = tf.shape(self.domain.X_f)
        self.X_f_len = tf.slice(self.X_f_dims, [0], [1]).numpy()
        tmp = [np.reshape(vec, (-1,1)) for i, vec in enumerate(self.domain.X_f.T)]
        self.X_f_in = np.asarray(tmp)
        self.u_model = neural_net(self.layer_sizes)



        if isAdaptive:
            self.isAdaptive = True
            if self.col_weights is None and self.u_weights is None:
                raise Exception("Adaptive weights selected but no inputs were specified!")
        if (
            not isAdaptive
            and self.col_weights is not None
            and self.u_weights is not None
        ):
            raise Exception(
                "Adaptive weights are turned off but weight vectors were provided. Set the weight vectors to "
                "\"none\" to continue")

    def compile_data(self, x, t, y):
        if not self.assimilate:
            raise Exception(
                "Assimilate needs to be set to 'true' for data assimilation. Re-initialize CollocationSolver1D with "
                "assimilate=True.")
        self.data_x = x
        self.data_t = t
        self.data_s = y

    def update_loss(self):
        loss_tmp = 0.0
        # Periodic BC iteration for all components of deriv_model
        for bc in self.bcs:
            if bc.isPeriodic:
                for i, dim in enumerate(bc.var):
                    for j, lst in enumerate(dim):
                        for k, tup in enumerate(lst):
                            upper = bc.u_x_model(self.u_model, bc.upper[i])[j][k]
                            lower = bc.u_x_model(self.u_model, bc.lower[i])[j][k]
                            msq = MSE(upper, lower)
                            loss_tmp = tf.math.add(loss_tmp, msq)
                continue
            # initial BCs, including adaptive model
            if bc.isInit:
                if self.isAdaptive:
                    loss_tmp = tf.math.add(loss_tmp, MSE(self.u_model(bc.input), bc.val, self.u_weights))
                else:
                    loss_tmp = tf.math.add(loss_tmp, MSE(self.u_model(bc.input), bc.val))
            # Dirichlect BC, will need to add more cases for Neumann BC, etc as more
            # BC types are added
            # This is true unless the BC loss can be evaluated using the MSE function explicitly
            else:
                loss_tmp = tf.math.add(loss_tmp, MSE(self.u_model(bc.input), bc.val))

        f_u_pred = self.f_model(self.u_model, *self.X_f_in)

        if self.isAdaptive:
            mse_f_u = MSE(f_u_pred, constant(0.0), self.col_weights)
        else:
            mse_f_u = MSE(f_u_pred, constant(0.0))

        loss_tmp = tf.math.add(loss_tmp, mse_f_u)
        return loss_tmp

    #@tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.update_loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    def fit(self, tf_iter = 0, newton_iter = 0, batch_sz=None, newton_eager=True):
        if self.isAdaptive and (batch_sz is not None):
            raise Exception("Currently we dont support minibatching for adaptive PINNs")
        if self.dist:
            fit_dist(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)
        else:
            fit(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)

    # L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(self):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value = self.update_loss()
            grad = tape.gradient(loss_value, self.u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def predict(self, X_star):
        X_star = convertTensor(X_star)
        u_star = self.u_model(X_star)

        f_u_star = self.f_model(self.u_model, X_star[:, 0:1],
                                X_star[:, 1:2])

        return u_star.numpy(), f_u_star.numpy()


# WIP
# TODO DiscoveryModel
class DiscoveryModel():
    def compile(self, layer_sizes, f_model, X, u, var, col_weights=None):
        self.layer_sizes = layer_sizes
        self.f_model = get_tf_model(f_model)
        self.X = X
        self.u = u
        self.vars = var
        self.len_ = len(var)
        self.u_model = neural_net(self.layer_sizes)
        self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_vars = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
        self.col_weights = col_weights
        #tmp = [np.reshape(vec, (-1,1)) for i, vec in enumerate(self.X)]
        self.X_in = tuple(X)
        #self.X_in = np.asarray(tmp).T
       # print(np.shape(self.X_in))

    @tf.function
    def loss(self):
        u_pred = self.u_model(tf.concat(self.X, 1))
        print(self.vars)
        f_u_pred = self.f_model(self.u_model, self.vars, *self.X_in)
        print(self.vars)

        if self.col_weights is not None:
            return MSE(u_pred, self.u) + g_MSE(f_u_pred, constant(0.0), self.col_weights ** 2)
        else:
            return MSE(u_pred, self.u) + MSE(f_u_pred, constant(0.0))

    @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    @tf.function
    def train_op(self):
        self.variables = self.u_model.trainable_variables
        len_ = self.len_
        if self.col_weights is not None:

            self.variables.extend([self.col_weights])
            self.variables.extend(self.vars)
            loss_value, grads = self.grad()
            self.tf_optimizer.apply_gradients(zip(grads[:-(len_ + 2)], self.u_model.trainable_variables))
            self.tf_optimizer_weights.apply_gradients(zip([-grads[-(len_ + 1)]], [self.col_weights]))
            self.tf_optimizer_vars.apply_gradients(zip(grads[-len_:], self.vars))
        else:
            self.variables.extend(self.vars)
            loss_value, grads = self.grad()

            self.tf_optimizer.apply_gradients(zip(grads[:-(len_ + 1)], self.u_model.trainable_variables))

            self.tf_optimizer_vars.apply_gradients(zip(grads[-len_:], self.vars))

        return loss_value

    def fit(self, tf_iter):
        self.train_loop(tf_iter)

    # @tf.function
    def train_loop(self, tf_iter):  # sourcery skip: move-assign
        start_time = time.time()
        for i in range(tf_iter):
            loss_value = self.train_op()
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f' % (i, elapsed))
                tf.print(f"loss_value: {loss_value}")
                var = [var.numpy() for var in self.vars]
                tf.print(f"vars estimate(s): {var}")
                start_time = time.time()
