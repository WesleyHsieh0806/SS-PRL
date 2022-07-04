import numpy as np
from sklearn.metrics import average_precision_score

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