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
    
    ds_tst = CustomImageDataset(os.path.join(annotations_file,training_mode + '_df.csv'), os.path.join(img_dir, training_mode + '_mode', training_mode), mode= 'test', transform = transform)

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

    # # 결과 출력
    # for e, a in zip(emotion, accuracies):
    #     print(f"Accuracy for {e}: {a:.4f}")

    # # 결과 출력
    # for e, a in zip(emotion, precision):
    #     print(f"precision for {e}: {a:.4f}")

    # AP score
    # print(ap_score)


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

    # # ap_score
    # plt.figure(figsize=(10, 8))
    # sns.set(font_scale=1.2)  # text size
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')

    # plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall Curve (AUC-PR={0:0.2f})'.format(auc_pr))
    # plt.show()

    
def get_args_parser(add_help=True):
    import argparse
  
    parser = argparse.ArgumentParser(description="Pytorch models test", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    test(config)

