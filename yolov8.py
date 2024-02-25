import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import yaml
import torch
from tqdm.auto import tqdm
import shutil
from PIL import Image

from ultralytics import YOLO


model = YOLO('yolov8n.pt')



            # epochs, batch, lr0
hype_list = [[10, 128, 0.0001], 
             [10, 128, 0.0001], 
             [10, 128, 0.0001], 
             [10, 128, 0.0001], 
             [10, 128, 0.0001],
             [10, 128, 0.0001],
             [10, 128, 0.0001],
             [10, 128, 0.0001],
             [10, 128, 0.0001],
             [10, 128, 0.0001]
                            ]

for epochs, batch, lr0 in hype_list:
    model.train(data="/home/KDT-admin/data/yolo/yolo.yaml",epochs=epochs,patience=5,batch=batch,optimizer='auto', # SGD, Adam, AdamW, NAdam, RAdam, RMSProp 등
                    lr0=lr0,imgsz=640)