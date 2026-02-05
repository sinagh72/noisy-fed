import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    Swin_V2_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights,
    ResNeXt101_64X4D_Weights, ViT_L_32_Weights, ConvNeXt_Large_Weights,
)
import torch.nn.functional as F

def build_backbone_model(model, pretrained=True, num_classes=10):
    if model == "resnet18":
        m = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        backbone = m

    elif model == "resnet50":
        m = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        backbone = m

    elif model == "resnext101_64x4d":
        m = torchvision.models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        backbone = m

    elif model == "swinv2_b":
        m = torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = m.head.in_features
        m.head = nn.Identity()
        backbone = m

    elif model == "swinv2_t":
        m = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = m.head.in_features
        m.head = nn.Identity()
        backbone = m

    elif model == "swinv2_s":
        m = torchvision.models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = m.head.in_features
        m.head = nn.Identity()
        backbone = m

    elif model == "vit_l_32":
        m = torchvision.models.vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None)
        # torchvision ViT: heads is the classifier
        if isinstance(m.heads, nn.Sequential):
            in_dim = m.heads[-1].in_features
        else:
            in_dim = m.heads.head.in_features
        m.heads = nn.Identity()
        backbone = m

    elif model == "convnext_large":
        m = torchvision.models.convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        # classifier is Sequential([... , Linear(in_dim,1000)])
        in_dim = m.classifier[-1].in_features
        m.classifier = nn.Identity()
        backbone = m
    
    elif model == "dinov3_small":
        m = torch.hub.load(
            repo_or_dir="./",
            model="dinov3_vits16",
            source="local",
            weights="/data/models/dinov3_vits16_pretrain_lvd1689m-8aa4cbdd.pth",
        )
        in_dim = m.embed_dim
        backbone = m

    elif model == "dinov3_base":
        m = torch.hub.load(
            repo_or_dir="./",
            model="dinov3_vitb16",
            source="local",
            weights="/data/models/dinov3_vitb16_pretrain_lvd1689m-8aa4cbdd.pth",
        )
        in_dim = m.embed_dim
        backbone = m

    elif model == "dinov3_large":
        m = torch.hub.load(
            repo_or_dir="./",
            model="dinov3_vitl16",
            source="local",
            weights="/data/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        )
        in_dim = m.embed_dim
        backbone = m

    else:
        raise ValueError(f"Unknown backbone: {model}")

    class Model(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone          # returns FEATURES now
            self.head = nn.Linear(in_dim, num_classes)

        def forward(self, x, return_feats=False):
            z = self.backbone(x)              # [B, in_dim]
            if isinstance(z, (tuple, list)):
                z = z[0]
            logits = self.head(z)
            if return_feats:
                return logits, z
            return logits

        @torch.no_grad()
        def extract_features(self, x):
            """
            which:
            - "backbone": returns [B, in_dim]
            """
            return self.backbone(x)
    
    return Model(backbone, num_classes)
