import torch
import torch.nn as nn
import torch.nn.functional as F

# img size = 48 * 48 전처리 필요, params 개수 체크한 후 너무 많으면 가벼운 모델로 만들기
class classifi_net(nn.Module):
    def __init__(self, num_classes=7):
        super(self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5)
        x = self.fc4(x)
        return F.softmax(x, dim=1)