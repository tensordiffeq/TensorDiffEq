import tensorflow as tf
import keras
import numpy as np
from .utils import *
from .networks import *
from .plotting import *

class CollocationModel1D:
    def __init__(self):
        self.layer_sizes = None #[2, 128, 128, 128, 128, 1]
        self.sizes_w = None
        self.sizes_b = None
        self.optimizer_NN = None
        self.u_model = None
        self.periodicBC = False
        self.X_f = None
        self.X0 = None
        self.u0 = None
        self.X_lb = None
        self.X_ub = None
        self.f_model = None
        self.u_x_model = None
        self.col_weights = None
        self.u_weights = None


    def compile(self, layer_sizes, f, X_f, X0, X_ub, X_lb, isPeriodic = False, u_x_model = None, isAdaptive = False, col_weights = None, u_weights = None):
        self.u_model = neural_net(layer_sizes)
        print("Network Architecture:")
        self.u_model.summary()
        self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.X0 = X0
        self.X_lb = X_lb
        self.X_ub = X_ub
        self.X_f = X_f
        self.f_model = get_tf_model(f)
        if isPeriodic:
            self.periodicBC = True
            if not self.u_x_model:
                print("Periodic BC is listed but no u_x model is defined!")
            else:
                self.u_x_model = get_tf_model(u_x_model)
        if isAdaptive:
            if not col_weights or u_weights:
                print("Adaptive weights selected but no inputs were specified!")
            else:
                self.col_weights = col_weights
                self.u_weights = u_weights


    def loss(self):
        f_u_pred = self.f_model(X_f)
        u0_pred = u_model(self.X0)

        if self.periodicBC:
            u_lb_pred, u_x_lb_pred = self.u_x_model(self.X_lb)
            u_ub_pred, u_x_ub_pred = self.u_x_model(self.X_ub)
            mse_b_u = MSE(u_lb_pred,u_ub_pred) + MSE(u_x_lb_pred, u_x_ub_pred)
        else:
            u_lb_pred = u_model(self.X_lb)
            u_ub_pred = u_model(self.X_ub)
            mse_b_u = MSE(u_lb_pred, self.u_lb) + MSE(u_ub_pred, self.u_ub)

        mse_0_u = MSE(self.u0, u0_pred, self.u_weights)

        mse_f_u = MSE(f_u_pred, constant(0.0), self.col_weights)

        return  mse_0_u + mse_b_u + mse_f_u , mse_0_u, mse_b_u, mse_f_u

    @tf.function
    def grad(self):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_b, mse_f = loss()
            grads = tape.gradient(loss_value, self.u_model.trainable_variables)
            grads_col = tape.gradient(loss_value, self.col_weights)
            grads_u = tape.gradient(loss_value, self.u_weights)

        return loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u


    def fit(self, tf_iter, newton_iter, batch_sz = None):

        #Can adjust batch size for collocation points, here we set it to N_f
        if batch_sz is not None:
            self.batch_sz = batch_sz
        else:
            self.batch_sz = len(self.X_f)

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

                X_f_batch = tf.slice(self.X_f, i*batch_sz, (i*batch_sz + batch_sz))

                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u = grad()

                tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
                tf_optimizer_weights.apply_gradients(zip([-grads_col, -grads_u], [col_weights, u_weights]))

            if epoch % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f' % (epoch, elapsed))
                tf.print(f"mse_0: {mse_0}  mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")
                start_time = time.time()

        #l-bfgs-b optimization
        print("Starting L-BFGS training")

        loss_and_flat_grad = get_loss_and_flat_grad()

        lbfgs(loss_and_flat_grad,
          get_weights(u_model),
          Struct(), maxIter=newton_iter, learningRate=0.8)


    #L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad():
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value, _, _, _ = loss()
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
