import numpy as np
from scipy import sparse

def enkf_update(
    data_f,
    data_m,
    n_ensemble,
    dim_obs,
    obs_var,
    obs_index,
):
    """
    EnKF update
    # modified based on https://link.springer.com/content/pdf/10.1007/s10236-003-0036-9.pdf
    Parameters
    ----------
    data_f : 2d array, Forecast state ensemble. (dim_vals, n_ensemble)
    data_m : 1d array, Measurement.
    n_ensemble : int, number of ensembles.
	dim_obs : int, number of observations.
	obs_var : float, measurement variance.
	obs_index : 1d array.
	
    Returns
    ----------
    data_a : 2d array,  Update state ensemble. (dim_vals, n_ensemble)
    """
    
    # Compute deviations from the mean
    A_mat_a = data_f - data_f.mean(axis=1)[:, np.newaxis]
    
    # Observation noise covariance matrix R (diagonal)
    # R_mat = sparse.diags(np.full(dim_obs, obs_var))
    # P_mat = (1/(n_ensemble-1))* (A_mat_a @ A_mat_a.T)

    # Generate observations perturbations
    Dobs_mat = np.zeros((dim_obs, n_ensemble))
    G_mat = np.zeros((dim_obs, n_ensemble))
    for i_real in range(n_ensemble):
        e_d = np.random.normal(0, 1, dim_obs)
        G_mat[:, i_real] = np.sqrt(obs_var) * e_d  
        Dobs_mat[:, i_real] = data_m + G_mat[:, i_real]  
        
    # Observation operator H
    H_mat = np.zeros((dim_obs, data_f.shape[0]))
    H_mat[np.arange(dim_obs), obs_index] = 1
    
    # Calculate observation anomalies
    D_mat_a = Dobs_mat - np.matmul(H_mat, data_f)
    
    X1_mat = np.matmul(H_mat, A_mat_a) + G_mat
    U, S, Vt = np.linalg.svd(X1_mat, full_matrices=False, compute_uv=True)
    
    # Construct the Kalman gain matrix
    Lambda = np.outer(S,S.T)
	K_gain = A_mat_a  @ (H_mat @ A_mat_a).T @ U @ (np.linalg.inv(Lambda) @ U.T )
    
    
    # Alternative way to check the matrix
#   X1 = np.linalg.inv(Lambda) @ U.T
# 	X2 = X1  @ D_mat_a
# 	X3 = U @ X2      
# 	X4 = (H_mat @ A_mat_a).T @ X3  
# 	data_a = data_f +  A_mat_a @ X4
    
    data_a = data_f + np.matmul(K_gain, D_mat_a)
    
    return data_a        
    