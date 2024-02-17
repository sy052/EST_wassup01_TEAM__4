import torch
import numpy as np 
 
 
class EarlyStopping(object):
    def __init__(self, patience=2, save_path="stopped_model.pth"):
        self._min_loss = np.inf
        self._patience = patience
        self._path = save_path
        self.__counter = 0
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False
   
    def load(self, model):
        model.load_state_dict(torch.load(self._path))
        return model
    
    @property
    def counter(self):
        return self.__counter