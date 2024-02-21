# 데이터셋 구성

# 트레인, 테스트
# 데이터 로더
# for image, label -> len in dataloader

# 모델 1 (mediapipe)
# 모델1(image) = ouput1

# 모델 2

# 모델2(output1)

# 이하 동일 


def test(cfg):
    from custom_dataset import CustomImageDataset
    from models.expression import Emotion_expression
    import os
    import torch
    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, average_precision_score, precision_recall_curve
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from deepface import DeepFace
    import cv2

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

    modelfile = '19140202' # find and input model num
    my_model.load_state_dict(torch.load(
        os.path.join(archive,
                    selected_model,
                    'pth', 'bt',
                    f'./{modelfile}_model.pth') 
    ))
    training_mode = cfg.get('training_mode')
    tst_dl_params = test_params.get('tst_data_loader_params')
    
    ds_tst = CustomImageDataset(os.path.join(annotations_file,training_mode + '_df.csv'), os.path.join(img_dir, training_mode + '_mode', training_mode), transform = transform)

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

            # detect model
            face = DeepFace.detectFace(img_path=inputs,
                    detector_backend='mediapipe')

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

            # AP
            


    # confusion matrix
    cm = confusion_matrix(cm_true, cm_pred)
    norm_cm = confusion_matrix(cm_true, cm_pred, normalize='true')

    per_cm = multilabel_confusion_matrix(cm_true, cm_pred)
    #ap_score = average_precision_score(np.array(cm_true).reshape(1, -1), np.array(cm_pred).reshape(1, -1))

    # precision, recall, _ = precision_recall_curve(cm_true, cm_pred)

    # auc_pr = average_precision_score(cm_true, cm_pred)
    
    # per_cm
    accuracies = []
    precision = []
    recall = []
    f1_score = []

    for i in range(len(emotion)):
        bcm = per_cm[i]  
        tp, tn, fp, fn = bcm.ravel()
        class_accuracy = (tp + tn) / (tp + tn + fp + fn)
        class_precision = tp / (tp + fp)
        class_recall = tp / (tp + fn)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)

        accuracies.append(class_accuracy)
        precision.append(class_precision)
        recall.append(class_recall)
        f1_score.append(class_f1)

    metrics_list = [(accuracies, "accuracies"), (precision, "precision"), (recall, "recall"), f1_score]

    for i in range(4):
        for e, metric in zip(emotion, metrics_list[i]):
            print(f"{metrics_list[i]} for {e}: {metric:.4f}")