import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # plt interactive mode

def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        
    return model

def visualize_model_predictions(model, img_path, device, class_names):
    was_training = model.training # 현재 트레이닝 모드 저장
    model.eval()

    img = Image.open(img_path)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        plt.imshow(img.cpu().data[0])

        model.train(mode=was_training) # 다시 이전 트레이닝 모드로 변경

def trainer(cfg):
    from custom_dataset import CustomImageDataset
    from models.expression import Emotion_expression

    # model_cfg
    model_cfg = cfg.get('model_cfg')
    model_list = model_cfg.get('model_list')
    choice_one = model_cfg.get('choice_one')
    selected = model_list[choice_one]
    selected_model = selected[0]
    model_weights =selected[1]
    
    path = cfg.get('path')
    annotations_file = path.get('annotations_dir')
    img_dir = path.get('img_dir')

    class_names = cfg.get('class_names')
    cls_len = len(class_names)

    train_params = cfg.get('train_params')
    device = train_params.get('device')
    loss_fn = train_params.get('loss_fn')
    optimizer = train_params.get('optim')
    epochs = train_params.get('epochs')
    

    # load pretrained model

    classifi = Emotion_expression(selected_model, cls_len)
    model_conv = classifi.model

    # preprocess
    transform = model_weights.transforms() # need before fitted model


    # Parameters of newly constructed modules have requires_grad=True by default
    #num_ftrs = model_conv.fc.in_features
    #model_conv.fc = nn.Linear(num_ftrs, 7) # input classes num
    #my_model = model_conv.fc.to(device) # 다른 변수명 필요

    
    my_model = model_conv.to(device)

    # low-level freeze
    freeze_percentage = cfg.get('freeze_percentage')
    freeze_line = len(list(my_model.parameters())) // freeze_percentage
    count = 0
    for param in my_model.parameters():
        if count > freeze_line:
            break
        param.requires_grad = False
        count += 1

   
    dl_params = train_params.get('data_loader_params')
    image_datasets = {x: CustomImageDataset(os.path.join(annotations_file, x + '_df.csv'), os.path.join(img_dir, x), transform = transform) # data_transforms[x] : option
                for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], **dl_params)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optim_params = train_params.get('optim_params')
    optimizer = optimizer(my_model.parameters(), **optim_params)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_model = train_model(my_model, dataloaders, dataset_sizes, device, loss_fn, optimizer, scheduler, epochs)

    # log

    log = path.get('output_log')
    torch.save(best_model.state_dict(), f'./{log}_model.pth')


    


    # # visualize
    # visualize_model_predictions(model_conv)

    # plt.ioff()
    # plt.show()

    # # traind_visualize
    # visualize_model_predictions(
    # model_conv,
    # img_path='data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg',
    # device=device,
    # class_names=class_names,
    # data_transforms = data_transforms
    # )

    # plt.ioff()
    # plt.show()

def get_args_parser(add_help=True):
    import argparse
  
    parser = argparse.ArgumentParser(description="Pytorch models trainer", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    trainer(config)