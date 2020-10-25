import numpy as np

def find_L2_error(u_pred, u_star):
    return np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
