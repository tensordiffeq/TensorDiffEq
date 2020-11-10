
import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *
from .fit import *

class CollocationSolver1D:
    def __init__(self, assimilate = False):
        self.sizes_w = None
        self.sizes_b = None
        self.optimizer_NN = None
        self.col_weights = None
        self.u_weights = None
        self.assimilate = assimilate


    def compile(self, layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, u_ub = None, u_lb = None, isPeriodic = False, u_x_model = None, isAdaptive = False, col_weights = None, u_weights = None, g = None, dist = False):
        self.layer_sizes = layer_sizes
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.x0 = x0
        self.t0 = t0
        self.u0 = u0
        self.x_lb = x_lb
        self.t_lb = t_lb
        self.u_lb = u_lb
        self.x_ub = x_ub
        self.t_ub = t_ub
        self.u_ub = u_ub
        self.x_f = x_f
        self.t_f = t_f
        self.f_model = get_tf_model(f_model)
        self.isAdaptive = False
        self.g = g
        self.dist = dist
        #self.u_x_model = get_tf_model(u_x_model)
        if isPeriodic:
            self.periodicBC = True
            if not u_x_model:
                raise Exception("Periodic BC is listed but no u_x model is defined!")
            else:
                self.u_x_model = get_tf_model(u_x_model)

        self.col_weights = col_weights
        self.u_weights = u_weights

        if isAdaptive:
            self.isAdaptive = True
            if self.col_weights is None and self.u_weights is  None:
                raise Exception("Adaptive weights selected but no inputs were specified!")
        if not isAdaptive:
            if self.col_weights is not None and self.u_weights is not None:
                raise Exception("Adaptive weights are turned off but weight vectors were provided. Set the weight vectors to \"none\" to continue")

    def compile_data(self, x, t, y):
        if not self.assimilate:
            raise Exception("Assimilate needs to be set to 'true' for data assimilation. Re-initialize CollocationSolver1D with assimilate=True.")
        self.data_x = x
        self.data_t = t
        self.data_s = y

    @tf.function
    def loss(self):
        if self.dist:
            f_u_pred = self.f_model(self.u_model, self.dist_x_f, self.dist_t_f)
        else:
            f_u_pred = self.f_model(self.u_model, self.x_f, self.t_f)
        u0_pred = self.u_model(tf.concat([self.x0, self.t0],1))

        if self.periodicBC:
            u_lb_pred, u_x_lb_pred = self.u_x_model(self.u_model, self.x_lb, self.t_lb)
            u_ub_pred, u_x_ub_pred = self.u_x_model(self.u_model, self.x_ub, self.t_ub)
            mse_b_u = MSE(u_lb_pred,u_ub_pred) + MSE(u_x_lb_pred, u_x_ub_pred)
        else:
            u_lb_pred = self.u_model(tf.concat([self.x_lb, self.t_lb],1))
            u_ub_pred = self.u_model(tf.concat([self.x_ub, self.t_ub],1))
            mse_b_u = MSE(u_lb_pred, u_lb) + MSE(u_ub_pred, u_ub)

        mse_0_u = MSE(u0_pred, self.u0, self.u_weights)

        if self.g is not None:
            if self.dist:
                mse_f_u = g_MSE(f_u_pred, constant(0.0), self.g(self.dist_col_weights))
            else:
                mse_f_u = g_MSE(f_u_pred, constant(0.0), self.g(self.col_weights))

        else:
            mse_f_u = MSE(f_u_pred, constant(0.0))

        if self.assimilate:
            s_pred = self.u_model(tf.concat([self.data_x, self.data_t],1))
            mse_s_u = MSE(s_pred, self.data_s)
            return mse_0_u + mse_b_u + mse_f_u + mse_s_u, mse_0_u, mse_b_u, mse_f_u
        else:
            return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

    def grad(self):
        with tf.GradientTape() as tape:
            loss_value, mse_0, mse_b, mse_f = self.loss()
            grads = tape.gradient(loss_value, self.variables)
        return loss_value, mse_0, mse_b, mse_f, grads


    def fit(self, tf_iter, newton_iter, batch_sz = None):
        if self.dist:
            fit_dist(self, tf_iter = tf_iter, newton_iter = newton_iter, batch_sz = batch_sz)
        else:
            fit(self, tf_iter = tf_iter, newton_iter = newton_iter, batch_sz = batch_sz)


    #L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(self):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value, _, _, _ = self.loss()
            grad = tape.gradient(loss_value, self.u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            #print(loss_value, grad_flat)
            return loss_value, grad_flat

        return loss_and_flat_grad


    def predict(self, X_star):
        X_star = convertTensor(X_star)
        u_star, _ = self.u_x_model(self.u_model, X_star[:,0:1],
                         X_star[:,1:2])

        f_u_star = self.f_model(self.u_model, X_star[:,0:1],
                     X_star[:,1:2])

        return u_star.numpy(), f_u_star.numpy()


class CollocationSolver2D(CollocationSolver1D):

    def compile(self, layer_sizes, f_model, x_f, y_f, t_f, x0, t0, u0, x_lb, y_lb, t_lb, x_ub, y_ub, t_ub, isPeriodic = False, u_x_model = None, isAdaptive = False, col_weights = None, u_weights = None, g = None):
        CollocationSolver1D.compile(layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isPeriodic, u_x_model, isAdaptive, col_weights, u_weights, g)
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.y_f = y_f

    def loss(self):
        f_u_pred = self.f_model(self.u_model, self.x_f, self.y_f, self.t_f)
        u0_pred = self.u_model(tf.concat([self.x0, self.y0, self.t0],1))

        u_lb_pred, u_x_lb_pred, u_y_lb_pred = self.u_x_model(self.u_model, self.x_lb, self.y_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred, u_y_ub_pred = self.u_x_model(self.u_model, self.x_ub, self.y_ub, self.t_ub)

        mse_b_u = MSE(u_lb_pred,u_ub_pred) + MSE(u_x_lb_pred, u_x_ub_pred) + MSE(u_y_lb_pred, u_y_ub_pred)

        mse_0_u = MSE(u0_pred, self.u0, self.u_weights)

        if self.g is not None:
            mse_f_u = g_MSE(f_u_pred, constant(0.0), self.g(self.col_weights))
        else:
            mse_f_u = MSE(f_u_pred, constant(0.0))

        return  mse_0_u + mse_b_u + mse_f_u , mse_0_u, mse_b_u, mse_f_u


class DiscoveryModel():
    def compile(self, layer_sizes, f_model, X, u, vars):
        self.layer_sizes = layer_sizes
        self.f_model = f_model
        self.X = X
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.u = u
        self.vars = vars
        self.u_model = neural_net(self.layer_sizes)
        self.tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        self.tf_optimizer_vars = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        print(self.x)

    def loss(self):
        u_pred = self.u_model(self.X)
        f_u_pred, self.vars = self.f_model(self.u_model, self.x, self.t, self.vars)

        return MSE(u_pred, self.u) + MSE(f_u_pred, constant(0.0))

    def fit(self, tf_iter):
        self.train_loop(tf_iter)

    @tf.function
    def grad(self):
        with tf.GradientTape(persistent = True) as tape:
            loss_value = self.loss()
            grads = tape.gradient(loss_value, self.u_model.trainable_variables)
            var_grads = tape.gradient(loss_value, self.vars)
            del tape
        return loss_value, grads, var_grads

    @tf.function
    def train_op(self):
        loss_value, grads_model, grads_vars = self.grad()
        self.tf_optimizer.apply_gradients(zip(grads_model, self.u_model.trainable_variables))
        self.tf_optimizer_vars.apply_gradients(zip(grads_vars, self.vars))
        return loss_value


    def train_loop(self, tf_iter):
        start_time = time.time()
        for i in range(tf_iter):
            loss_value = self.train_op()
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f' % (i, elapsed))
                tf.print(f"total loss: {loss_value}")
                var = [var.numpy() for var in self.vars]
                print("vars estimate(s):", var)
                start_time = time.time()
