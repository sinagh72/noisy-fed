import torch.nn as nn
import torchvision
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    Swin_V2_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights,
    ResNeXt101_64X4D_Weights, ViT_L_32_Weights, ConvNeXt_Large_Weights,
)

def build_backbone_model(model, pretrained=True, num_classes=10):
    """
    Keeps the backbone's original classifier (fc/head/classifier) and adds a new
    linear head on TOP of the backbone output (usually 1000-d ImageNet logits).

    Returns:
        nn.Module: forward(x) -> logits [B, num_classes]
    """

    # ----------------- build torchvision model as-is -----------------
    if model == "resnet18":
        backbone = torchvision.models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = backbone.fc.out_features  # typically 1000

    elif model == "resnet50":
        backbone = torchvision.models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = backbone.fc.out_features  # typically 1000

    elif model == "swinv2_b":
        backbone = torchvision.models.swin_v2_b(
            weights=Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = backbone.head.out_features  # typically 1000

    elif model == "swinv2_t":
        backbone = torchvision.models.swin_v2_t(
            weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = backbone.head.out_features

    elif model == "swinv2_s":
        backbone = torchvision.models.swin_v2_s(
            weights=Swin_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = backbone.head.out_features

    elif model == "resnext101_64x4d":
        backbone = torchvision.models.resnext101_64x4d(
            weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_dim = backbone.fc.out_features  # typically 1000

    elif model == "vit_l_32":
        backbone = torchvision.models.vit_l_32(
            weights=ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # vit heads is a Sequential; last module is usually Linear(?,1000)
        last = backbone.heads[-1] if isinstance(backbone.heads, nn.Sequential) else backbone.heads.head
        if not isinstance(last, nn.Linear):
            raise TypeError("Unexpected ViT heads structure; expected last layer to be nn.Linear.")
        in_dim = last.out_features  # typically 1000

    elif model == "convnext_large":
        backbone = torchvision.models.convnext_large(
            weights=ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # classifier is Sequential(..., Linear(?,1000)) as last element
        last = backbone.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("Unexpected ConvNeXt classifier structure; expected last layer to be nn.Linear.")
        in_dim = last.out_features  # typically 1000

    else:
        raise ValueError(f"Unknown backbone: {model}")

    # ----------------- wrapper: backbone logits -> new head -----------------
    class BackbonePlusHead(nn.Module):
        def __init__(self, backbone, in_dim, num_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(in_dim, num_classes)

        def forward(self, x):
            y = self.backbone(x)  # typically [B, 1000]
            if isinstance(y, (tuple, list)):
                y = y[0]
            return self.head(y)

    return BackbonePlusHead(backbone, in_dim, num_classes)
