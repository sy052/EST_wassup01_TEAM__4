class Emotion_expression:
    def __init__(self, model_name, cls_len, weights:str = "DEFAULT"):
        self.model_name = model_name
        self.cls_len = cls_len
        self.model_weights = weights
        self.model = self.build_model()
        

    def build_model(self):
        from torchvision.models import get_model
        import torch.nn as nn

        from models.custom_cls import classifi_net

        if self.model_name == 'custom':
            model = classifi_net()
            
        else:
            model = get_model(self.model_name, weights=self.model_weights)

            if self.model_name == 'alexnet':
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'convnext_tiny':
                num_ftrs = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(num_ftrs, self.cls_len)

            elif self.model_name == 'densenet121':
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, self.cls_len)

            elif self.model_name == 'efficientnet_v2_s':
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'googlenet':
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.cls_len)

            elif self.model_name == 'inception_v3':
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'mnasnet0_5':
                num_ftrs = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'mobilenet_v3_large':
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'mobilenetv3':
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = nn.Linear(num_ftrs, self.cls_len)

            elif self.model_name == 'resnet18' or self.model_name == 'resnet34' or self.model_name == 'resnet50':
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'resnext50_32x4d':
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'vgg11' or self.model_name == 'vgg13_bn' or self.model_name == 'vgg16_bn':
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, self.cls_len)
            
            elif self.model_name == 'vit_b_16':
                num_ftrs = model.heads[0].in_features
                model.heads[0] = nn.Linear(num_ftrs, self.cls_len)

            elif self.model_name == 'swin_t':
                num_ftrs = model.head.in_features
                model.head = nn.Linear(num_ftrs, self.cls_len)

        
        return model