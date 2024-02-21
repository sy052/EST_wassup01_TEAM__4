import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from ultralytics import YOLO
from PIL import Image

class CustomImageDataset(Dataset):
    # annotations_file : labels path, img_dir : images folder path
    def __init__(self, annotations_file:str, img_dir:str, mode:str='train', transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['file_name', 'class'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.face_detection_model = YOLO('yolov8n.pt')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0][:-4]+'.jpg')
        image = Image.open(img_path)

        if self.mode == 'test':
            image = []
            img = cv2.imread(os.path.join(data_path, img_name))
            result = self.face_detection_model.predict(source=img, conf=0.6, half=False)
            result = result[0].cpu().numpy()

            for box in result.boxes:
                xyxy = box.xyxy[0].astype(int).tolist()
                x1, y1, x2, y2 = xyxy
                image.append(img[y1:y2, x1:x2])
            
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label