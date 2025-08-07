import torch
import torchvision

class GeminioResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super(GeminioResNet18, self).__init__()

        self.upsample = torch.nn.Upsample(size=(224, 224), mode='bilinear')
        self.extractor = torchvision.models.resnet18(weights='IMAGENET1K_V1')
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

class GeminioResNetModel(torch.nn.Module):
    def __init__(self, resnet_version='resnet18', num_classes=10):
        super(GeminioResNetModel, self).__init__()
        
        if resnet_version == 'resnet18':
            self.model = GeminioResNet18(num_classes)
        elif resnet_version == 'resnet34':
            self.model = GeminioResNet34(num_classes)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        # Expose classifier for convenience
        self.clf = self.model.clf

    def forward(self, x, return_features=False):
        return self.model(x, return_features)

class GeminioViTB16(torch.nn.Module):
    def __init__(self, num_classes):
        super(GeminioViTB16, self).__init__()

        self.upsample = torch.nn.Upsample(size=(224, 224), mode='bilinear')
        self.extractor = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(self.extractor.heads[0].in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
        self.extractor.heads = torch.nn.Identity()

    def forward(self, x, return_features=False):
        x = self.upsample(x)
        features = self.extractor(x)
        outputs = self.clf(features)
        if return_features:
            return features, outputs
        return outputs