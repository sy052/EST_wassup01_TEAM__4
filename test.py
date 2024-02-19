def test(cfg):
    from custom_dataset import CustomImageDataset
    from models.expression import Emotion_expression
    import os
    import torch
    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
    from collections import defaultdict

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
    transform = model_weights.transforms() # need before fitted model

    my_model = model_conv.to(device)

    modelfile = '' # find and input model num
    my_model.load_state_dict(torch.load(
        os.path.join(archive, 
                    selected_model,
                    'pth', 'bt',
                    f'./{modelfile}_model.pth') 
    ))

    tst_dl_params = test_params.get('tst_data_loader_params')
    tst_dataset_sizes = len(dl_tst.dataset)
    tst_dl_params['batch_size'] = tst_dataset_sizes
    training_mode = cfg.get('training_mode')

    ds_tst = CustomImageDataset(os.path.join(annotations_file,training_mode + '_df.csv'), os.path.join(img_dir, training_mode + '_mode', training_mode), transform = transform)
    dl_tst = torch.utils.data.DataLoader(ds_tst, **tst_dl_params)
    loss_fn = train_params.get('loss_fn')

    labels = ['angry', 'anxiety', 'embarrass', 'happy', 'normal', 'pain', 'sad']
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


    # total
    epoch_loss = running_loss / tst_dataset_sizes
    epoch_acc = running_corrects.float() / tst_dataset_sizes

    # confusion matrix
    cm = confusion_matrix(cm_true, cm_pred, labels=labels)
    norm_cm = confusion_matrix(cm_true, cm_pred, labels=labels, normalize='all')

    per_cm = multilabel_confusion_matrix(cm_true, cm_pred, labels=labels)
    

    
    
def get_args_parser(add_help=True):
    import argparse
  
    parser = argparse.ArgumentParser(description="Pytorch models test", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    test(config)

