import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR
from datetime import datetime

config = {

  'files': {
    'annotations_file': '/home/KDT-admin/data/labels',
    'img_dir' : '/home/KDT-admin/data/images'

  },
  'output': {
    'output_log': datetime.now().strftime("%d%H%M%S"),
    'load_pth_24':'../../results/ann/best24.pth',
    'load_pth_168':'../../results/ann/best168.pth',
    'load_pth_168d':'../../results/ann/best24d.pth',
  },
  'preprocess_params': {
  },
  'class_names' : {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
  },
  'freeze_percentage': 80,
  'model': MANN,
  
  'train_params': {
    'dataset_params':{
    },
    'data_loader_params': {
      'batch_size': 4,
      'shuffle': True,
      'num_workers': 4
    },
    'loss_fn': nn.functional.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.001,
      'weight_decay': 0
    },

    'lr_scheduler1': ReduceLROnPlateau,
    'scheduler_params1': {
      'mode': 'min',
      'factor': 0.1,
      'patience': 5,
      'verbose':False
    },

    'lr_scheduler2': CosineAnnealingWarmRestarts,
    'scheduler_params2': {
      'T_0': 30,
      'T_mult': 3,
      'eta_min': 0.00001,
      'last_epoch':-1,
      'verbose':False
    },
    
    'lr_scheduler3': CyclicLR,
    'scheduler_params3': {
      'base_lr': 0.0000001,
      'max_lr': 0.001,
      'step_size_up': 15,
      'mode': "triangular2",
      'gamma':0.55,
      'cycle_momentum':False
    },
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'epochs': 200,
  },
}