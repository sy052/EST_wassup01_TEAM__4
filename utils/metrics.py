import numpy as np
import pandas as pd
def cm_to_metrics(cm, emotion):
    df = pd.DataFrame(columns=["accuracy", "precision", "recall"])

    for i in range(cm.shape[0]):
        accuracy_i = (cm[i, i] + np.sum(np.diag(cm)) - cm[i, i]) / np.sum(cm)
        precision_i = cm[i, i] / (np.sum(cm[:, i]) + 1e-8)
        recall_i = cm[i, i] / (np.sum(cm[i, :]) + 1e-8)

        
        df.loc[i] = [accuracy_i, precision_i, recall_i]
    
    df.index = emotion
    df.loc["total"] = df.mean(axis=0)
    
    return df