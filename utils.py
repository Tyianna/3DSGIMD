import os
import random
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from scipy.stats import spearmanr

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cal_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    ci = cal_ci(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    pearson_value = np.corrcoef(y_true, y_pred)[0, 1]
    spearman_value, _ = spearmanr(y_true, y_pred)
    d = {'mae': mae, 'ci': ci, 'mse': mse, 'rmse': rmse, 'r2': r2, 'pearson_value':pearson_value, 'spearman_value':spearman_value}
    return d

def binary_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    auc = roc_auc_score(y_true, y_score)
    if y_pred is None: y_pred = (y_score >= threshod).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prauc = metrics.auc(precision_recall_curve(y_true, y_score)[1], precision_recall_curve(y_true, y_score)[0])
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    BAS = balanced_accuracy_score(y_true, y_pred)

    d = {'auc': auc, 'prauc': prauc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'BAS':BAS}
    return d
    
def binary_metrics_multi_target_nan(y_true, y_score, y_pred=None, nan_fill=-1, threshod=0.5):
    '''
       y_true and y_score should be `(N, T)` where N = number of SAMELs, and T = number of targets
    '''
    if y_pred is None: y_pred = (y_score >= threshod).astype(int)
    roc_list, acc_list, prc_list, rec_list = [], [], [], []
    for i in range(y_true.shape[1]):
        if (y_true[:, i] == 1).sum() == 0 or (y_true[:, i] == 0).sum() == 0:
            print('Skipped target, cause AUC is only defined when there is at least one positive data.')
            continue

        if nan_fill == -1:
            is_valid = y_true[:, i] >= 0
            y_true_st = y_true[is_valid, i]
            y_score_st = y_score[is_valid, i]
            y_pred_st = y_pred[is_valid, i]
            roc_list.append(roc_auc_score(y_true_st, y_score_st))
            acc_list.append(accuracy_score(y_true_st, y_pred_st))
            prc_list.append(precision_score(y_true_st, y_pred_st))
            rec_list.append(recall_score(y_true_st, y_pred_st))
    d = {'auc': sum(roc_list) / len(roc_list), 'acc': sum(acc_list) / len(acc_list),
         'precision': sum(prc_list) / len(prc_list), 'recall': sum(rec_list) / len(rec_list)}
    return d
