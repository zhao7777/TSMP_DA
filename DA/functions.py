'''
EnKF function(s)
'''
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


def enkf_update(
    fullensDF,
    num_ensemble,
    dim_vals,
    obs_ens,
    dim_obs,
    obs_var,
    obs_index,
    save_covariance_matrix=False,
    save_correlation_matrix=False,
    visualize=False,
    outname="output",
):
    """
    EnKF update

    # BASIC EnKF equations:
    # pmat = 1/(n-1) * sum_i=1^N ( (x^i - mean(x))(x^i - mean(x))^t
    # Rmat = diag(0.02)
    # H = 1 where obs is 0 else
    # kgain = pmat * Ht * ( Rmat + H*P*Ht)^-1
    # xana = xcurr + kgain * ( obs - hx)
    # update = before - after (= xcurr - xana)

    Parameters
    ----------
    fullensDF : pd.DataFrame
        Forecast state ensemble.
        Dimensions: (dim_vals, num_ensemble)

    num_ensemble : int
        Ensemble size.

    dim_vals : int
        State vector size.

    obs_ens : np.ndarray
        Observation ensemble.
        Dimensions: (dim_obs, num_ensemble)

    dim_obs : int
        Observation vector size.

    obs_var : float
        Observation variance.

    obs_index : list
        Indices of observation variables in state vector.
        Used for defining the measurement operator.

    save_covariance_matrix : bool
        Switch for saving the covariance matrix of the
        update to file named ``pmat.npy``.

    save_correlation_matrix : bool
        Switch for saving the correlation matrix of the
        update to file named ``cmat.npy``. Compared to saving the
        covariance matrix, this adds the computational demand for
        computing the correlation matrix.

    visualize : bool
        Switch for output of figure of correlations.

    outname : str
        Filename.

    Returns
    ----------
    xanaDF : pd.DataFrame
        Update state ensemble.
        Dimensions: (dim_vals, num_ensemble)
    """

    # mean_val_DF = fullensDF.mean(axis=1)
    # mean_val = np.array(mean_val_DF)

    # Covariance matrix
    pmat_DF = fullensDF.T.cov()
    pmat = np.array(pmat_DF)

    # Optionally save the correlation matrix to `pmat.npy`
    if save_covariance_matrix:
        np.save("pmat.npy", pmat)

    # Optionally compute and save the correlation matrix to `cmat.npy`
    if save_correlation_matrix:
        cmat_DF = fullensDF.T.corr()
        cmat = np.array(cmat_DF)
        np.save("cmat.npy", cmat)

    # Optionally plot the covariance matrix with a contour plot
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(pmat, interpolation="nearest")
        fig.colorbar(cax)
        plt.savefig(outname + "correlation_matrix.png", dpi=180)
        plt.close()

    # rmat assumed to be iid so diag of observation variance

    rmat = np.diag(obs_var * np.ones(dim_obs))

    # hmat assumed to be zero except where observations are
    hmat = np.zeros((dim_obs, dim_vals))
    for i in range(0, dim_obs):
        hmat[i, obs_index[i]] = 1
    print(hmat)
    # hmat[0, obs_index[0]] = 1

    # kalman gain as shown in basic EnKF equation using numpy matrix
    # multiplications
    kgain = np.matmul(
        np.matmul(pmat, hmat.transpose()),
        np.linalg.inv(
            (rmat + np.matmul(np.matmul(hmat, pmat), hmat.transpose()))))
    
    print(kgain)
    # update calculation and application
    xanaDF = (fullensDF +
              np.matmul(kgain[:, :], obs_ens - np.matmul(hmat, fullensDF)))

    return xanaDF
