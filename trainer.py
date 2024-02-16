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
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

def visualize_model_predictions(model,img_path, device, class_names, data_transforms=None):
    was_training = model.training # 트레이닝 된 것을 불러오자. 코드 수정 필요
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        plt.imshow(img.cpu().data[0])

        model.train(mode=was_training)

def trainer(cfg):
    from custom_dataset import CustomImageDataset

    # load pretrained model
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    # low-level freeze
    freeze_percentage = cfg.get('freeze_percentage')
    freeze_line = len(model_conv.parameters()) // freeze_percentage
    count = 0
    for param in model_conv.parameters():
        if count > freeze_line:
            break
        param.requires_grad = False
        count += 1

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 7) # input classes num
    model_conv = model_conv.fc.to(device) # 다른 변수명 필요

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    files = cfg.get('files')
    annotations_file = files.get('annotations_file')
    img_dir = files.get('img_dir')

    class_names = class_names.get('class_names')

    dl_params = train_params.get('data_loader_params')
    image_datasets = {x: CustomImageDataset(os.path.join(annotations_file, x), os.path.join(img_dir, x)) # data_transforms[x] : option
                for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], **dl_params)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    train_params = cfg.get('train_params')
    device = train_params.get('device')
    loss_fn = train_params.get('loss_fn')
    optimizer = train_params.get('optim')
    epochs = train_params.get('epochs')
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optim_params = train_params.get('optim_params')
    optimizer = optimizer(model_conv.fc.parameters(), **optim_params)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    traind_model = train_model(model_conv, dataloaders, dataset_sizes, device, loss_fn, optimizer, scheduler, epochs)


    # visualize
    visualize_model_predictions(model_conv)

    plt.ioff()
    plt.show()

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