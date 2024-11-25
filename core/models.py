import torch
import torchvision

class GeminioResNet34(torch.nn.Module):
    def __init__(self, num_classes):
        super(GeminioResNet34, self).__init__()

        self.upsample = torch.nn.Upsample(size=(224, 224), mode='bilinear')
        self.extractor = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(self.extractor.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
        self.extractor.fc = torch.nn.Identity()

    def forward(self, x, return_features=False):
        x = self.upsample(x)
        features = self.extractor(x)
        outputs = self.clf(features)
        if return_features:
            return features, outputs
        return outputs