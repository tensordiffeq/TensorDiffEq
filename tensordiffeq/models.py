import tensorflow as tf
import keras
import numpy as np
import time
from .utils import *
from .networks import *
from .plotting import *

class CollocationModel1D:
    def __init__(self):
        self.sizes_w = None
        self.sizes_b = None
        self.optimizer_NN = None
        #self.f_model = None
        #self.u_x_model = None
        self.col_weights = None
        self.u_weights = None


    def compile(self, layer_sizes, f, x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, isPeriodic = False, u_x_model = None, isAdaptive = False, col_weights = None, u_weights = None):
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
        # #self.f_model = get_tf_model(self.f_model)
        # if isPeriodic:
        self.periodicBC = True
        #     if not u_x_model:
        #         raise Exception("Periodic BC is listed but no u_x model is defined!")
        #     #else:
        #         #self.u_x_model = get_tf_model(u_x_model)

        self.col_weights = col_weights
        self.u_weights = u_weights
        # if isAdaptive:
        #     if self.col_weights == None and self.u_weights == None:
        #         raise Exception("Adaptive weights selected but no inputs were specified!")


    @tf.function
    def f_model(self, x, t):
        u = self.u_model(tf.concat([x, t],1))
        u_x = tf.gradients(u, x)
        u_xx = tf.gradients(u_x, x)
        u_t = tf.gradients(u,t)
        c1 = constant(.0001)
        c2 = constant(5.0)
        f_u = u_t - c1*u_xx + c2*u*u*u - c2*u
        return f_u

    @tf.function
    def u_x_model(self, x, t):
        u = self.u_model(tf.concat([x, t],1))
        print(x)
        u_x = tf.gradients(u, x)
        print(u_x)
        return u, u_x


    def loss(self):
        u = self.u_model(tf.concat([self.x_f, self.t_f],1))
        f_u_pred = self.f_model(self.x_f, self.t_f)
        u0_pred = self.u_model(tf.concat([self.x0, self.t0],1))

        u_lb_pred, u_x_lb_pred = self.u_x_model(self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.x_ub, self.t_ub)
        #print(u_x_lb_pred)
        mse_b_u = tf.reduce_mean(tf.square(tf.math.subtract(u_lb_pred,u_ub_pred))) + \
            tf.reduce_mean(tf.square(tf.math.subtract(u_x_lb_pred, u_x_ub_pred)))
        mse_0_u = tf.reduce_mean(tf.square(self.u_weights*(self.u0 - u0_pred)))

        mse_f_u = tf.reduce_mean(self.col_weights**2*tf.square(f_u_pred))

        return  mse_0_u + mse_b_u + mse_f_u , mse_0_u, mse_b_u, mse_f_u

    @tf.function
    def grad(self):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_b, mse_f = self.loss()
            grads = tape.gradient(loss_value, self.u_model.trainable_variables)
            grads_col = tape.gradient(loss_value, self.col_weights)
            grads_u = tape.gradient(loss_value, self.u_weights)

        return loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u


    def fit(self, tf_iter, newton_iter, batch_sz = None):

        #Can adjust batch size for collocation points, here we set it to N_f
        if batch_sz is not None:
            self.batch_sz = batch_sz
        else:
            self.batch_sz = len(self.x_f)

        N_f = len(self.x_f)
        n_batches =  N_f // self.batch_sz

        start_time = time.time()
        #create optimizer s for the network weights, collocation point mask, and initial boundary mask
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        tf_optimizer_weights = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        tf_optimizer_u = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)

        print("starting Adam training")

        # For mini-batch (if used)
        for epoch in range(tf_iter):
            for i in range(n_batches):

                #X_f_batch = tf.slice(self.X_f, i*self.batch_sz, (i*self.batch_sz + self.batch_sz))

                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u = self.grad()

                tf_optimizer.apply_gradients(zip(grads, self.u_model.trainable_variables))
                tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u], [self.col_weights, self.u_weights]))

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


    def predict(X_star):
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
        u_star, _ = u_x_model(X_star[:,0:1],
                         X_star[:,1:2])

        f_u_star = f_model(X_star[:,0:1],
                     X_star[:,1:2])

        return u_star.numpy(), f_u_star.numpy()



    # Define constants and weight vectors
