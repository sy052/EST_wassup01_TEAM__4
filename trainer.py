import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from collections import defaultdict

cudnn.benchmark = True
plt.ion()   # plt interactive mode

def train_one_epoch(model, dataloaders,device, criterion, optimizer):
    model.train()  # Set model to training mode  # Set model to evaluate mode

    dataset_sizes = len(dataloaders.dataset)
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.item() / dataset_sizes

    return epoch_loss, epoch_acc

def evaluate(model, dataloaders, device, criterion):
    model.eval()   # Set model to evaluate mode

    dataset_sizes = len(dataloaders.dataset)
    running_loss = 0.0
    running_corrects = 0
    with torch.inference_mode():
    # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.item() / dataset_sizes

    return epoch_loss, epoch_acc

def visualize_model_predictions(model, img_path, device, class_names):
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

def trainer(cfg):
    from tqdm.auto import trange

    from custom_dataset import CustomImageDataset
    from models.expression import Emotion_expression
    from utils.earlystopping import EarlyStopping

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
    log = path.get('output_log')
    archive = path.get('archive')

    training_mode = cfg.get('training_mode')
    class_names = cfg.get('class_names')
    cls_len = len(class_names)

    train_params = cfg.get('train_params')
    test_params = cfg.get('test_params')
    device = train_params.get('device')
    loss_fn = train_params.get('loss_fn')
    optimizer = train_params.get('optim')

    earlystopping = train_params.get('earlystopping')
    patience = earlystopping.get('patience')
    stop_path = os.path.join(archive, 
                            selected_model,
                            'pth', 'st',
                            f'./{log}_stopped.pth')

    early_stopper = EarlyStopping(patience, stop_path)

    
    

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

    trn_dl_params = train_params.get('trn_data_loader_params')
    tst_dl_params = test_params.get('tst_data_loader_params')
    
    
    
    if training_mode == 'val':
        ds_trn = CustomImageDataset(os.path.join(annotations_file,'train_df.csv'), os.path.join(img_dir,training_mode + '_mode' ,'train'), transform = transform)
        ds_tst = CustomImageDataset(os.path.join(annotations_file,training_mode + '_df.csv'), os.path.join(img_dir, training_mode + '_mode', training_mode), transform = transform)


    elif training_mode == 'test':
        ds_trn = CustomImageDataset(os.path.join(annotations_file,'train_df.csv'), os.path.join(img_dir,training_mode + '_mode' ,'train'), transform = transform)
        ds_tst = CustomImageDataset(os.path.join(annotations_file,training_mode + '_df.csv'), os.path.join(img_dir, training_mode + '_mode', training_mode), transform = transform)

    # tst batch_size
    tst_dataset_sizes = len(ds_tst)
    tst_dl_params['batch_size'] = tst_dataset_sizes

    dl_trn = torch.utils.data.DataLoader(ds_trn, **trn_dl_params)
    dl_tst = torch.utils.data.DataLoader(ds_tst, **tst_dl_params)

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optim_params = train_params.get('optim_params')
    optimizer = optimizer(my_model.parameters(), **optim_params)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    pbar = trange(train_params.get('epochs'))
    history = defaultdict(list)

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(my_model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for i in pbar:
            trn_loss, trn_acc = train_one_epoch(my_model, dl_trn,device, loss_fn, optimizer)
            tst_loss, tst_acc = evaluate(my_model, dl_tst, device, loss_fn)
            scheduler.step()

            history['trn_loss'].append(trn_loss)
            history['trn_acc'].append(trn_acc)
            history['tst_loss'].append(tst_loss)
            history['tst_acc'].append(tst_acc)

            if tst_acc > best_acc:
                best_acc = tst_acc
                torch.save(my_model.state_dict(), best_model_params_path)
            
            # early_stopping
            if early_stopper.should_stop(my_model, trn_loss):
                print(f"EarlyStopping: [Epoch: {i+1 - early_stopper.counter}]")
                break
            
            pbar.set_postfix(trn_loss=trn_loss, tst_loss=tst_loss, Tst_acc= tst_acc)
        
        # load best model weights
        my_model.load_state_dict(torch.load(best_model_params_path))
        # best model save
        torch.save(my_model.state_dict(), os.path.join(archive, 
                                                     selected_model,
                                                     'pth', 'bt',
                                                     f'./{log}_model.pth'))
    




    ###########
    ### log ###
    ###########
    
    

   

    # loss
    y1 = history['trn_loss']
    y2 = history['tst_loss']

    tst_min = min(history['tst_loss'])
    min_idx = history['tst_loss'].index(tst_min)

    plt.figure(figsize=(8, 6))
    plt.plot(y1, color='#16344E', label='trn_loss')
    plt.plot(y2, color='#71706C', label='tst_loss')
    plt.legend()
    plt.title(f"{selected_model} Losses, Min_loss(test):{tst_min:.4f}, Min_idx(test):{min_idx+1}")
    plt.savefig(os.path.join(archive, 
                            selected_model,
                            'png',
                            f'./{log}_losses.png'))
    
    # accuracy
    


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