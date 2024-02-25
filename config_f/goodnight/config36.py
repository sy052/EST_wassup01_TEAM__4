import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR
from datetime import datetime
import pytz
from torchvision import models

config = {

  'path': {
    'annotations_dir': '/home/KDT-admin/data_2000/labels',
    'img_dir' : '/home/KDT-admin/data_2000/images/cropped',
    'output_log': datetime.now(pytz.timezone('Asia/Seoul')).strftime("%d%H%M%S"),
    'archive': '/home/KDT-admin/work/soyeon/EST_wassup01_TEAM__4/archive',
    'pkl_path_list_trn': [
      ('/home/KDT-admin/work/bonghoon/EST_wassup01_TEAM__4/0_v_ds_trn.pkl',
       '/home/KDT-admin/work/bonghoon/EST_wassup01_TEAM__4/0_v_ds_tst.pkl'),

    ],
    'pkl_path_list_tst' : [

    ]
  },
  
  'training_mode': 'test', # choose between val and test
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
  
  'freeze_percentage': 0.9, # 10 단위로 변경
  'model_cfg': {
    'choice_one' : 0,
    'model_list': [
        ['alexnet', models.AlexNet_Weights.IMAGENET1K_V1], # 0 
        ['convnext_tiny', models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1], # 1
        ['densenet121', models.DenseNet121_Weights.IMAGENET1K_V1], # 2
        ['efficientnet_v2_s', models.EfficientNet_V2_S_Weights.IMAGENET1K_V1], # 3
        ['googlenet', models.GoogLeNet_Weights.IMAGENET1K_V1],# 4
        ['inception_v3', models.Inception_V3_Weights.IMAGENET1K_V1], # 5
        ['mnasnet0_5', models.MNASNet0_5_Weights.IMAGENET1K_V1], # 6
        ['mobilenet_v3_small', models.MobileNet_V3_Small_Weights.IMAGENET1K_V1], # 7
        ['resnet18', models.ResNet18_Weights.IMAGENET1K_V1], # 8
        ['resnet34', models.ResNet34_Weights.IMAGENET1K_V1], # 9
        ['resnet50', models.ResNet50_Weights.IMAGENET1K_V1], # 10
        ['resnext50_32x4d', models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1], # 11
        ['vgg11_bn', models.VGG11_BN_Weights.IMAGENET1K_V1], # 12
        ['vgg13_bn', models.VGG13_BN_Weights.IMAGENET1K_V1], # 13
        ['vgg16_bn', models.VGG16_BN_Weights.IMAGENET1K_V1], # 14
        ['vit_b_16', models.ViT_B_16_Weights.IMAGENET1K_V1], # 15
        ['swin_t', models.Swin_T_Weights.IMAGENET1K_V1], # 16
    ]
  },
  'test_params': {
    'tst_data_loader_params': {
      'batch_size': "Auto",
      'shuffle': False,
      'num_workers': 4
    }
  },
  'train_params': {
    'device': torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu'),
    'epochs': 70,
    
    'earlystopping':{
      'patience': 9,
    },

    'trn_data_loader_params': {
      'batch_size': 256,
      'shuffle': True,
      'num_workers': 4
    },
    
    'loss_fn': nn.functional.cross_entropy,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.0001,
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
    
  },
}