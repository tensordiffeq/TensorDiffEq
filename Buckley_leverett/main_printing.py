from Analytical import u_analytical
import pickle
from utils_buckley_leverett import *

# hprc_jobID = ['55901', '55902', '55903', '55904a', '55904b', '55914','55924']
hprc_jobID = ['56560', '56572', '56796']

for hprc_jobID in hprc_jobID:
    if hprc_jobID != '':
        hprc_jobID = '_' + hprc_jobID

    ## Open file with results
    with open(fr'../Results/output{hprc_jobID}.pickle', 'rb') as f:
        predictions, losses, Ms, xt, case_tests = pickle.load(f)

    X, T = np.meshgrid(*xt)
    for case, preds in predictions.items():
        # Analytical solutions
        Exact_u = u_analytical(case, Ms[case][0], X.flatten(), T.flatten())

        subcases = list(preds.keys())
        preds_list = list(preds.values())

        u_pred = np.hstack(list(zip(*preds_list))[0])
        f_u_pred = np.hstack(list(zip(*preds_list))[1])

        lb = [0.0, 0.0]  # x_inf, t_inf
        ub = [1.0, 1.0]  # x_sup, t_sup

        ## Plot losses
        # tf.config.run_functions_eagerly(True)
        loss = losses[case]
        epochs = [len(loss_i) for loss_i in loss.values()]
        epoch_adam_std = min(epochs)
        epoch_lbfgs_std = max(epochs)

        # plot_losses(loss, title=case.capitalize() + hprc_jobID, divider=epoch_adam_std,
        #             xlim=epoch_adam_std + epoch_lbfgs_std)

        ## Plot predictions
        plot_solution_domain1D_v2([u_pred, f_u_pred], xt,
                                  ub=ub, lb=lb, Title=case + hprc_jobID, Legends=subcases,
                                  Exact_u=Exact_u)

        ##

        for j, u_pred in enumerate(u_pred.T):
            error_u = find_L2_error(u_pred, Exact_u)
            print(f'Error {subcases[j]} : {error_u:.3e}')
