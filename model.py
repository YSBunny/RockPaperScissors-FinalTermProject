import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.vgg(x)
        return out
