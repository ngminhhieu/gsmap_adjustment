import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import scipy.signal
from matplotlib import pyplot as plt

def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


cnn_gt = pd.read_csv('./results/correlation/conv2d/default/groundtruth.csv').to_numpy()
cnn_pd = pd.read_csv('./results/correlation/conv2d/default/preds.csv').to_numpy()
cnn_gt = np.transpose(cnn_gt)
cnn_pd = np.transpose(cnn_pd)
cnn_corr = []
for i in range(cnn_gt.shape[1]):
    cnn_corr.append(np.corrcoef((cnn_pd[:,i], cnn_gt[:,i]))[0][1])

cnn_corr = np.array(cnn_corr)

lstm_gt = pd.read_csv('./log/ed_lstm/groundtruth.csv').to_numpy()
lstm_pd = pd.read_csv('./log/ed_lstm/preds.csv').to_numpy()
lstm_corr = []
for i in range(lstm_gt.shape[1]):
    lstm_corr.append(np.corrcoef((lstm_pd[:,i], lstm_gt[:,i]))[0][1])

lstm_corr = np.array(lstm_corr)

print(nashsutcliffe(cnn_pd, cnn_gt))
print(nashsutcliffe(lstm_pd, lstm_gt))
np.savetxt("cnn_corr.csv", cnn_corr, delimiter=',')
np.savetxt("lstm_corr.csv", lstm_corr, delimiter=',')


