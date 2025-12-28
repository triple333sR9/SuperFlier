"""Model architectures for Bird Flight Classifier"""

import torch.nn as nn
from torchvision import models


def get_resnet50(pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    return model, "layer4"


def get_efficientnet_b3(pretrained=True):
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1), nn.Sigmoid())
    return model, "features.7"


def get_densenet121(pretrained=True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1), nn.Sigmoid())
    return model, "denseblock4"


def get_convnext_small(pretrained=True):
    weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
    model = models.convnext_small(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.LayerNorm(in_features), nn.Dropout(0.3), nn.Linear(in_features, 1), nn.Sigmoid()
    )
    return model, "features.7"


def get_resnext50(pretrained=True):
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
    model = models.resnext50_32x4d(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    return model, "layer4"


MODEL_REGISTRY = {
    "resnet50": get_resnet50,
    "efficientnet_b3": get_efficientnet_b3,
    "densenet121": get_densenet121,
    "convnext_small": get_convnext_small,
    "resnext50": get_resnext50,
}


def get_model(name, pretrained=True):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](pretrained)


def freeze_backbone(model, unfreeze_layer):
    for name, param in model.named_parameters():
        param.requires_grad = unfreeze_layer in name or "fc" in name or "classifier" in name