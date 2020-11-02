import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *

class CollocationSolver1D:
    def __init__(self):
        self.sizes_w = None
        self.sizes_b = None
        self.optimizer_NN = None
        self.col_weights = None
        self.u_weights = None


    def compile(self, layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isPeriodic = False, u_x_model = None, isAdaptive = False, col_weights = None, u_weights = None, g = None):
        self.u_model = neural_net(layer_sizes)
        print("Network Architecture:")
        self.u_model.summary()
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.x0 = x0
        self.t0 = t0
        self.u0 = u0
        self.x_lb = x_lb
        self.t_lb = t_lb
        self.x_ub = x_ub
        self.t_ub = t_ub
        self.x_f = x_f
        self.t_f = t_f
        self.f_model = get_tf_model(f_model)
        self.isAdaptive = False
        self.g = g
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


    def loss(self):
        f_u_pred = self.f_model(self.u_model, self.x_f, self.t_f)
        u0_pred = self.u_model(tf.concat([self.x0, self.t0],1))

        u_lb_pred, u_x_lb_pred = self.u_x_model(self.u_model, self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.u_model, self.x_ub, self.t_ub)

        mse_b_u = MSE(u_lb_pred,u_ub_pred) + MSE(u_x_lb_pred, u_x_ub_pred)

        mse_0_u = MSE(u0_pred, self.u0, self.u_weights)

        if self.g is not None:
            mse_f_u = g_MSE(f_u_pred, constant(0.0), self.g(self.col_weights))
        else:
            mse_f_u = MSE(f_u_pred, constant(0.0))

        return  mse_0_u + mse_b_u + mse_f_u , mse_0_u, mse_b_u, mse_f_u

    @tf.function
    def adaptgrad(self):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_b, mse_f = self.loss()
            grads = tape.gradient(loss_value, self.u_model.trainable_variables)
            grads_col = tape.gradient(loss_value, self.col_weights)
            grads_u = tape.gradient(loss_value, self.u_weights)
            del tape
        return loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u

    @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value, mse_0, mse_b, mse_f = self.loss()
            grads = tape.gradient(loss_value, self.u_model.trainable_variables)
            del tape
        return loss_value, mse_0, mse_b, mse_f, grads

    def fit(self, tf_iter, newton_iter, batch_sz = None):

        #Can adjust batch size for collocation points, here we set it to N_f
        if batch_sz is not None:
            self.batch_sz = batch_sz
        else:
            self.batch_sz = len(self.x_f)

        N_f = len(self.x_f)
        n_batches =  N_f // self.batch_sz

        start_time = time.time()
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        tf_optimizer_weights = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        #tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)

        print("starting Adam training")

        for epoch in range(tf_iter):
            for i in range(n_batches):
                if self.isAdaptive:
                    loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u = self.adaptgrad()
                    tf_optimizer.apply_gradients(zip(grads, self.u_model.trainable_variables))
                    tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u], [self.col_weights, self.u_weights]))
                else:
                    loss_value, mse_0, mse_b, mse_f, grads = self.grad()
                    tf_optimizer.apply_gradients(zip(grads, self.u_model.trainable_variables))

            if epoch % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f' % (epoch, elapsed))
                tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
                start_time = time.time()

        #l-bfgs-b optimization
        print("Starting L-BFGS training")

        loss_and_flat_grad = self.get_loss_and_flat_grad()

        lbfgs(loss_and_flat_grad,
          get_weights(self.u_model),
          Struct(), maxIter=newton_iter, learningRate=0.8)


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


class CollocationModel2D(CollocationModel1D):

    def compile(self, layer_sizes, f_model, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isPeriodic = False, u_x_model = None, isAdaptive = False, col_weights = None, u_weights = None, g = None):
        self.u_model = neural_net(layer_sizes)
        print("Network Architecture:")
        self.u_model.summary()
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.x0 = x0
        self.t0 = t0
        self.u0 = u0
        self.x_lb = x_lb
        self.y_lb = y_lb
        self.t_lb = t_lb
        self.x_ub = x_ub
        self.y_ub - y_ub
        self.t_ub = t_ub
        self.x_f = x_f
        self.y_f = y_f
        self.t_f = t_f
        self.f_model = get_tf_model(f_model)
        self.isAdaptive = False
        self.g = g
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
