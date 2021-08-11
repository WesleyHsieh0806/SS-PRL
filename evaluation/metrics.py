import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,precision_score,
    recall_score
)

def compute_AP(predictions, labels):
    predictions = predictions.T
    labels = labels.T
    num_class = predictions.shape[0]
    APs = []
    for i in range(num_class):
        mask = labels[i] <= 1
        AP = average_precision_score(labels[i][mask], predictions[i][mask]-1e-5*labels[i][mask])
        APs.append(AP)
    mAP = np.mean(APs)
    return APs, mAP

def compute_F1(predictions, labels, mode_F1):
    if mode_F1 == 'overall':
        print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = np.sum(labels[mask]==1)
        p = TP/np.sum(mask)
        r = TP/np.sum(labels==1)
        f1 = 2*p*r/(p+r)

    else:
        num_class = predictions.shape[1]
        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r  = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:,idx_cls])
            label = np.squeeze(labels[:,idx_cls])
            if np.sum(label>0) == 0:
                continue
            f1[idx_cls] = f1_score(label, prediction)
            p[idx_cls] = precision_score(label, prediction)
            r[idx_cls] = recall_score(label, prediction)

    return f1, p, r

    idx = np.argsort(predictions,axis = 1)
    for i in range(predictions.shape[0]):
        predictions[i][idx[i][-k:]]=1
        predictions[i][idx[i][:-k]]=0
