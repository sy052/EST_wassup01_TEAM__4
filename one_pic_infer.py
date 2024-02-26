from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import numpy as np
import torch
import opencv

def one_pic_inference(target_image):
    model = YOLO('archive/models/yolo/pth/best.pt')

    image_array = np.array(target_image)

    image_dtype = image_array.dtype
    
    with torch.no_grad():
        results = model.predict(image_array, conf = 0.5)
    
    for result in results:
        boxes = result.boxes

    class_names = model.names
    
    
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image_array)
    
    emotion = []

    # Iterate over the boxes and annotations
    for box, conf, class_idx in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
        # Get the class index
        class_index = int(class_idx)

        # Get the class name
        class_name = class_names[class_index]
        emotion.append(class_name)
        # Get the box coordinates
        x1, y1, x2, y2 = box[:4]

        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle patch to the axes
        ax.add_patch(rect)

        # Add the label and confidence score
        label = f"{class_name}: {conf:.2f}"
        ax.text(x1, y1 - 10, label, fontsize=15, color='b')

    return fig, ax, emotion[0]