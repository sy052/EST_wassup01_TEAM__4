import numpy as np
import pandas as pd
def cm_to_metrics(cm, emotion):
    df = pd.DataFrame(columns=["accuracy", "precision", "recall"])

    for i in range(cm.shape[0]):
        # accuracy_i = (cm[i, i] + np.sum(np.diag(cm)) - cm[i, i]) / np.sum(cm)
        # precision_i = cm[i, i] / (np.sum(cm[:, i]) + 1e-8)
        # recall_i = cm[i, i] / (np.sum(cm[i, :]) + 1e-8)
        
        true_positive = cm[i, i]
        false_positive = np.sum(cm[:, i]) - true_positive
        false_negative = np.sum(cm[i, :]) - true_positive
        true_negative = np.sum(cm) - (true_positive + false_positive + false_negative)

        accuracy_i = true_positive / (true_positive + false_negative + 1e-8)
        precision_i = true_positive / (true_positive + false_positive + 1e-8)
        recall_i = cm[i, i] / (np.sum(cm[i, :]) + 1e-8)

        
        df.loc[i] = [accuracy_i, precision_i, recall_i]
    
    df.index = emotion
    df.loc["total"] = df.mean(axis=0)
    
    return df