import numpy as np
import pandas as pd
def cm_to_metrics(cm, emotion):
    df = pd.DataFrame(columns=["accuracy", "precision", "recall"])

    for i in range(cm.shape[0]):
        # accuracy_i = (cm[i, i] + np.sum(np.diag(cm)) - cm[i, i]) / np.sum(cm)
        # precision_i = cm[i, i] / (np.sum(cm[:, i]) + 1e-8)
        # recall_i = cm[i, i] / (np.sum(cm[i, :]) + 1e-8)
        
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        accuracy_i = tp / (tp + fn + 1e-8)
        precision_i = tp / (tp + fp + 1e-8)
        recall_i = cm[i, i] / (np.sum(cm[i, :]) + 1e-8)

        
        df.loc[i] = [accuracy_i, precision_i, recall_i]
    
    df.index = emotion
    df.loc["total"] = df.mean(axis=0)
    
    return df

def cm_mento_metrics(cm, emotion):
    global df
    
    df = pd.DataFrame(columns=["accuracy", "precision", "recall"])
    

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        print(tp)
        tn = np.sum(np.diag(cm)) - tp
        print(f'tn:{tn}')
        fn = np.sum(cm[i]) - tp
        print(fn)
        # fp = np.sum(cm[:, i]) - tp + np.sum(cm[:])
        fp = np.sum(cm) - tp - tn - fn
        # fn = np.sum(cm[i, :]) - tp
        # tn = np.sum(cm) - (tp + fp + fn)

        accuracy_i = (tp + tn) / (tp + tn + fp + fn + 1e-8)  # Accuracy 계산 수정
        precision_i = tp / (tp + fp + 1e-8)
        recall_i = tp / (tp + fn + 1e-8)  # Recall 계산 수정
        
        df.loc[i] = [accuracy_i, precision_i, recall_i]
    
    df.index = emotion
    print(df)
    df.loc["total"] = df.mean(axis=0)
    
    return df