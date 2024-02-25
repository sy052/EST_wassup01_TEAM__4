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
import torchvision.transforms as transforms


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


def trainer(cfg):
    from tqdm.auto import trange
    import pickle

    from custom_dataset import CustomImageDataset
    from models.expression import Emotion_expression
    from utils.earlystopping import EarlyStopping

    # model_cfg
    model_cfg = cfg.get('model_cfg')
    model_list = model_cfg.get('model_list')
    choice_one = model_cfg.get('choice_one')
    selected = model_list[choice_one]
    selected_model = selected[0]
    model_weights = selected[1]
    
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
    torch.cuda.set_device(device) # change allocation of current GPU
    torch.cuda.manual_seed_all(777)

    loss_fn = train_params.get('loss_fn')
    optimizer = train_params.get('optim')

    earlystopping = train_params.get('earlystopping')
    patience = earlystopping.get('patience')

    early_stopper = EarlyStopping(patience)

    
    # load pretrained model

    classifi = Emotion_expression(selected_model, cls_len)
    model_conv = classifi.model

    my_model = model_conv.to(device)

    # preprocess
    if choice_one == 17:
        transform = transforms.Compose([
        transforms.Resize((48, 48), antialias=True),
        transforms.ToTensor()
        ])
    else:
        transform = model_weights.transforms(antialias=True) # need before fitted model
        
        # low-level freeze
        freeze_percentage = cfg.get('freeze_percentage')
        freeze_line = int(len(list(my_model.parameters())) * freeze_percentage) # default = 0.6
        count = 0
        for param in my_model.parameters():
            if count > freeze_line:
                break
            param.requires_grad = False
            count += 1
    

    trn_dl_params = train_params.get('trn_data_loader_params')
    tst_dl_params = test_params.get('tst_data_loader_params')
    
    
    if training_mode == 'val':
        ds_trn = CustomImageDataset(os.path.join(annotations_file,'v_trn_df.csv'), os.path.join(img_dir, training_mode + '_mode' ,'train'), mode = 'train',transform = transform)
        ds_tst = CustomImageDataset(os.path.join(annotations_file,'v_tst_df.csv'), os.path.join(img_dir, training_mode + '_mode', 'test'), mode = 'train', transform = transform)


    elif training_mode == 'test':
        ds_trn = CustomImageDataset(os.path.join(annotations_file,'t_trn_df.csv'), os.path.join(img_dir,training_mode + '_mode' ,'train'), mode = 'train', transform = transform)
        ds_tst = CustomImageDataset(os.path.join(annotations_file,'t_tst_df.csv'), os.path.join(img_dir, training_mode + '_mode', 'test'), mode = 'train', transform = transform)


    # with open(f"{choice_one}_v_ds_trn.pkl", "wb") as f:
    #     pickle.dump(ds_trn, f)
    
    # with open(f"{choice_one}_v_ds_tst.pkl", "wb") as f:
    #     pickle.dump(ds_tst, f)


    # if training_mode == 'val':
    #     pkl_path = pkl_path_list_trn[choice_one]
    #     trn_path  = pkl_path[0]
    #     tst_path  = pkl_path[1]

    #     with open(trn_path, "rb") as f:
    #         ds_trn = pickle.load(f)
    
    #     with open(tst_path, "rb") as f:
    #         ds_tst = pickle.load(f)
    
    # elif training_mode == 'test':
    #     pkl_path = pkl_path_list_trn[choice_one]
    #     trn_path  = pkl_path[2]
    #     tst_path  = pkl_path[3]

    #     with open(trn_path, "rb") as f:
    #         ds_trn = pickle.load(f)
    
    #     with open(tst_path, "rb") as f:
    #         ds_tst = pickle.load(f)

    if training_mode == 'val':
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
    early_stop_epoch = 0

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
            if early_stopper.should_stop(my_model, tst_loss):
                stop_epoch = i+1 - early_stopper.counter
                print(f"EarlyStopping: [Epoch: {stop_epoch}]")
                early_stop_epoch = stop_epoch
                break
            
            pbar.set_postfix(trn_loss=trn_loss, tst_loss=tst_loss, Tst_acc= tst_acc)
        
        # load best model weights
        my_model.load_state_dict(torch.load(best_model_params_path))
        # best model save
        torch.save(my_model.state_dict(), os.path.join(archive, 'models',
                                                     selected_model,
                                                     'pth', 'bt',
                                                     f'./{log}_model.pth'))
    




    ###########
    ### log ###
    ###########
    
    # loss
    y1 = history['trn_loss']
    y2 = history['tst_loss']

    tst_loss_min = min(history['tst_loss'])
    loss_min_idx = history['tst_loss'].index(tst_loss_min)

    plt.figure(figsize=(8, 6))
    plt.plot(y1, color='#16344E', label='trn_loss')
    plt.plot(y2, color='#71706C', label='tst_loss')
    plt.legend()
    plt.title(f"{selected_model} Losses, Min_loss(test):{tst_loss_min:.4f}, Min_idx(test):{loss_min_idx+1}")
    plt.savefig(os.path.join(archive, 'models',
                            selected_model,
                            'png',
                            f'./{log}_losses.png'))
    
    plt.close()
    # accuracy
    y1 = history['trn_acc']
    y2 = history['tst_acc']

    tst_acc_max = max(history['tst_acc'])
    acc_max_idx = history['tst_acc'].index(tst_acc_max)

    plt.figure(figsize=(8, 6))
    plt.plot(y1, color='#16344E', label='trn_acc')
    plt.plot(y2, color='#71706C', label='tst_acc')
    plt.legend()
    plt.title(f"{selected_model} accuracy, max(test):{tst_acc_max:.4f}, max_idx(test):{acc_max_idx+1}")
    plt.savefig(os.path.join(archive, 'models',
                            selected_model,
                            'png',
                            f'./{log}_accuracy.png'))
    
    return log, early_stop_epoch, history['tst_acc'][-1], tst_acc_max, acc_max_idx, history['trn_loss'][-1], history['tst_loss'][-1], tst_loss_min, loss_min_idx


def get_args_parser(add_help=True):
    import argparse
  
    parser = argparse.ArgumentParser(description="Pytorch models trainer", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime
    import pytz

    set_mode = 'onetime' # onetime, manytime
    many_mode = 'goodnight' # day_10, day_20, day_30, goodnight

    if set_mode == 'onetime':
        args = get_args_parser().parse_args()
        exec(open(args.config).read())
        trainer(config)

    elif set_mode == 'manytime':
        
        columns = ['training_mode', 'model_num', 'optim', 'image(EA)', 'freeze', 'epochs', 'batch','lr','weight_decay','PNG_NAME', 'early_stop_epoch', 'Tst_acc', 'Max_acc', 'Max_idx', 'trn_loss', 'tst_loss', 'Min_loss', 'Min_idx']
        df = pd.DataFrame(columns=columns)
        now = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%d%H%M%S")

        target_dir = os.path.join('/home/KDT-admin/work/soyeon/EST_wassup01_TEAM__4/config_f', many_mode)
        cfg_list = sorted(os.listdir(target_dir))

        for cfg in cfg_list:
            with open(os.path.join(target_dir, cfg), 'r') as f:
                config = f.read()
                exec(config)
                log, early_stop_epoch, Tst_acc, Max_acc, Max_idx, trn_loss, tst_loss, Min_loss, Min_idx = trainer(config)

                train_params = config.get('train_params')
                trn_data_loader_params = train_params.get('trn_data_loader_params')
                optim_params = train_params.get('optim_params')
                model_cfg = config.get('model_cfg')

                new_data = {
                    'training_mode': config.get('training_mode'),
                    'model_num': model_cfg.get('choice_one'),
                    'optim': str(train_params.get('optim')),
                    'image(EA)': 2000,
                    'freeze': config.get('freeze_percentage'),
                    'epochs': train_params.get('epochs'),
                    'batch': trn_data_loader_params.get('batch_size'),
                    'lr': optim_params.get('lr'),
                    'weight_decay': optim_params.get('weight_decay'),
                    'PNG_NAME': log,
                    'early_stop_epoch': early_stop_epoch,
                    'Tst_acc': round(Tst_acc, 4),
                    'Max_acc': round(Max_acc, 4),
                    'Max_idx': Max_idx,
                    'trn_loss': round(trn_loss, 4),
                    'tst_loss': round(tst_loss, 4),
                    'Min_loss': round(Min_loss, 4),
                    'Min_idx': Min_idx
                }
                df = pd.concat([df, pd.DataFrame([new_data], columns=columns)], ignore_index=True)

        root_dir = '/home/KDT-admin/work/soyeon/EST_wassup01_TEAM__4/archive/train_log'
        folder_name = now
        os.makedirs(os.path.join(root_dir, folder_name))
        save_dir = os.path.join(root_dir, folder_name)
        df.to_csv(os.path.join(save_dir, 'result.csv'), index=False)