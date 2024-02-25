def test(cfg):
    from custom_dataset import CustomImageDataset
    from models.expression import Emotion_expression
    from utils.metrics import cm_to_metrics
    import os
    import torch
    from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torchvision.transforms as transforms
    
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

    class_names = cfg.get('class_names')
    cls_len = len(class_names)

    test_params = cfg.get('test_params')
    train_params = cfg.get('train_params')
    
    
    device = train_params.get('device')

    classifi = Emotion_expression(selected_model, cls_len)
    model_conv = classifi.model

    # preprocess
    if choice_one == 17:
        transform = transforms.Compose([
        transforms.Resize((48, 48), antialias=True),
        transforms.ToTensor()
        ])
    else:
        transform = model_weights.transforms(antialias=True)

    my_model = model_conv.to(device)

    modelfile = '21154513' # find and input model num
    my_model.load_state_dict(torch.load(
        os.path.join(archive, 'models',
                    selected_model,
                    'pth', 'bt',
                    f'{modelfile}_model.pth') 
    ))
    training_mode = cfg.get('training_mode')
    tst_dl_params = test_params.get('tst_data_loader_params')
    
    #ds_tst = CustomImageDataset(os.path.join(annotations_file,'v_tst_df.csv'), os.path.join(img_dir, training_mode + '_mode', 'test'), mode = 'train', transform = transform)
    ds_tst = CustomImageDataset(os.path.join('/home/KDT-admin/data/sample_test_data/sample_df.csv'), 
                                os.path.join('/home/KDT-admin/data/sample_test_data'), mode = 'test', transform = transform)
    
    tst_dataset_sizes = len(ds_tst)
    tst_dl_params['batch_size'] = tst_dataset_sizes
    dl_tst = torch.utils.data.DataLoader(ds_tst, **tst_dl_params)

    loss_fn = train_params.get('loss_fn')

    emotion = ['angry', 'anxiety', 'embarrass', 'happy', 'normal', 'pain', 'sad']
    cm_pred = []
    cm_true = []
    

    my_model.eval()   # Set model to evaluate mode
    

    running_loss = 0.0
    running_corrects = 0

    with torch.inference_mode():
    # Iterate over data.
        for inputs, labels in dl_tst:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            outputs = my_model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # cm
            cm_pred += preds.data.cpu().tolist()
            cm_true += labels.data.cpu().tolist()

        total_acc = running_corrects.item() / tst_dataset_sizes
        
        # AP
            


    # confusion matrix
    cm = confusion_matrix(cm_true, cm_pred)
    norm_cm = confusion_matrix(cm_true, cm_pred, normalize='true')
    
    emotion_df = cm_to_metrics(cm, emotion)
    
    print(emotion_df)
    
    
    
    #ap_score = average_precision_score(np.array(cm_true).reshape(1, -1), np.array(cm_pred).reshape(1, -1))

    # precision, recall, _ = precision_recall_curve(cm_true, cm_pred)

    # auc_pr = average_precision_score(cm_true, cm_pred)
        
   
    
    # # precision_recall_curve
    # for i in range(cls_len):
    #     precision, recall, _ = precision_recall_curve(cm_true, cm_pred, pos_label=i)
    #     plt.plot(recall, precision, label=class_names[i])
    #     plt.legend(loc="lower left")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig('precision_recall_curve.png')


    # cm save
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)  # text size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion, yticklabels=emotion)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2) 
    sns.heatmap(norm_cm, annot=True, cmap='Blues', xticklabels=emotion, yticklabels=emotion)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_2.png')

    
def get_args_parser(add_help=True):
    import argparse
  
    parser = argparse.ArgumentParser(description="Pytorch models test", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    test(config)

