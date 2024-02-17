class Emotion_expression:
    def __init__(self, model_name, cls_len, weights:str = "DEFAULT"):
        self.model_name = model_name
        self.cls_len = cls_len
        self.model_weights = weights
        self.model = self.build_model()
        

    def build_model(self):
        from torchvision.models import get_model
        import torch.nn as nn

        model = get_model(self.model_name, weights=self.model_weights)

        if self.model_name == 'alexnet':
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.cls_len)

        return model