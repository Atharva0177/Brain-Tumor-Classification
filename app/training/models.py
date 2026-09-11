from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    convnext_base,
    convnext_tiny,
    efficientnet_b0,
    resnet50,
)

ARCHITECTURES = ("convnext_tiny", "convnext_base", "resnet50", "efficientnet_b0")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int = 4
    pretrained: bool = True


def _weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    return {
        "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_base": ConvNeXt_Base_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
    }[name]


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.name not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture {spec.name!r}; choose one of {ARCHITECTURES}")
    weights = _weights(spec.name, spec.pretrained)
    if spec.name in ("convnext_tiny", "convnext_base"):
        constructor = convnext_tiny if spec.name == "convnext_tiny" else convnext_base
        model = constructor(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, spec.num_classes)
    elif spec.name == "resnet50":
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, spec.num_classes)
    else:
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, spec.num_classes)
    return model


def classifier_parameters(model: nn.Module, architecture: str):
    if architecture in ("convnext_tiny", "convnext_base"):
        return model.classifier.parameters()
    if architecture == "resnet50":
        return model.fc.parameters()
    if architecture == "efficientnet_b0":
        return model.classifier.parameters()
    raise ValueError(f"Unsupported architecture {architecture!r}")


def freeze_for_head_warmup(model: nn.Module, architecture: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier_parameters(model, architecture):
        parameter.requires_grad = True


def unfreeze_last_stages(model: nn.Module, architecture: str, depth: int = 2) -> None:
    if depth < 1:
        raise ValueError("unfreeze depth must be at least 1")
    if architecture == "resnet50":
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif architecture in ("convnext_tiny", "convnext_base"):
        stages = list(model.features[1::2])
    elif architecture == "efficientnet_b0":
        stages = list(model.features)
    else:
        raise ValueError(f"Unsupported architecture {architecture!r}")
    for parameter in classifier_parameters(model, architecture):
        parameter.requires_grad = True
    for stage in stages[-depth:]:
        for parameter in stage.parameters():
            parameter.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
