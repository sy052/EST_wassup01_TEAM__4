import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR
from datetime import datetime
from torchvision import models

config = {

  'path': {
    'annotations_dir': '/home/KDT-admin/data/labels',
    'img_dir' : '/home/KDT-admin/data/images/cropped',
    'output_log': datetime.now().strftime("%d%H%M%S"),
    'archive': '/home/KDT-admin/work/bonghoon/EST_wassup01_TEAM__4/archive', 
    'pth_dir':'/home/KDT-admin/work/bonghoon/EST_wassup01_TEAM__4/archive',
    'png_dir' : '/home/KDT-admin/work/bonghoon/EST_wassup01_TEAM__4/archive'
  },
  
  'training_mode': 'val', # choose between val and test
  'class_names' : {
    0: "anger",
    1: "anxiety",
    2: "embrrass",
    3: "happy",
    4: "normal",
    5: "pain",
    6: "sad",
    # 7: "pain2",
  },
  'freeze_percentage': 80,
  'model_cfg': {
    'choice_one' : 0,
    'model_list': [
        ['alexnet', models.AlexNet_Weights.IMAGENET1K_V1],
        'convnext_tiny',
        'convnext_small',
        'densenet121',
    ]
  },
  
  'train_params': {
    'earlystopping':{
      'patience': 7,
    },
    'data_loader_params': {
      'batch_size': 4,
      'shuffle': True,
      'num_workers': 4
    },
    'loss_fn': nn.functional.cross_entropy,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.00001,
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
    'epochs': 2,
  },
}