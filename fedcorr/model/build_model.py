from model.lenet import LeNet
from model.model_resnet import ResNet18, ResNet34
from model.model_resnet_official import ResNet50
import torchvision.models as models
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights, Swin_V2_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, ResNeXt101_64X4D_Weights, ViT_L_32_Weights, ConvNeXt_Large_Weights

def build_model(args):
    # choose different Neural network model for different args or input
    if args.model == 'lenet':
        netglob = LeNet().to(args.device)

    elif args.model == 'resnet18':
        netglob = ResNet18(args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'resnet34':
        netglob = ResNet34(args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'resnet50':
        netglob = ResNet50(pretrained=False)
        if args.pretrained:
            model = models.resnet50(pretrained=True)
            netglob.load_state_dict(model.state_dict())
        netglob.fc = nn.Linear(2048, args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, args.num_classes)
        netglob = netglob.to(args.device)

    else:
        exit('Error: unrecognized model')

    return netglob


import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    Swin_V2_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights,
    ResNeXt101_64X4D_Weights, ViT_L_32_Weights, ConvNeXt_Large_Weights
)

# ------------------------- feature extractors -------------------------

class ResNetFeatureExtractor(nn.Module):
    """Returns penultimate features (e.g., 512 for resnet18, 2048 for resnet50)."""
    def __init__(self, arch="resnet18", pretrained=True):
        super().__init__()
        if arch == "resnet18":
            m = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            out_dim = m.fc.in_features  # 512
        elif arch == "resnet50":
            m = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            out_dim = m.fc.in_features  # 2048
        else:
            raise ValueError(f"ResNetFeatureExtractor only supports resnet18/resnet50, got {arch}")

        # remove final fc
        self.features = nn.Sequential(*list(m.children())[:-1])  # up to avgpool
        self.out_dim = out_dim

    def forward(self, x):
        z = self.features(x)      # [B, D, 1, 1]
        z = z.flatten(1)          # [B, D]
        return z


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_feature_backbone(model_architecture, pretrained=True):
    """
    Returns: (feature_extractor_module, feat_dim)
    Feature extractor must output [B, D] embeddings (NOT logits).
    """
    if model_architecture in ("resnet18", "resnet50"):
        feat = ResNetFeatureExtractor(model_architecture, pretrained=pretrained)
        return feat, feat.out_dim

    # --- Swin V2 ---
    if model_architecture == "swinv2_b":
        m = torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.head.in_features
        m.head = Identity()   # output becomes [B, feat_dim]
        return m, feat_dim

    if model_architecture == "swinv2_t":
        m = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.head.in_features
        m.head = Identity()
        return m, feat_dim

    if model_architecture == "swinv2_s":
        m = torchvision.models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.head.in_features
        m.head = Identity()
        return m, feat_dim

    # --- ResNeXt ---
    if model_architecture == "resnext101_64x4d":
        m = torchvision.models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = Identity()
        return m, feat_dim

    # --- ViT ---
    if model_architecture == "vit_l_32":
        m = torchvision.models.vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.heads.head.in_features
        m.heads = Identity()  # output becomes [B, feat_dim]
        return m, feat_dim

    # --- ConvNeXt ---
    if model_architecture == "convnext_large":
        m = torchvision.models.convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        # convnext classifier is Sequential(LayerNorm2d, Flatten, Linear)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = Identity()  # output becomes [B, feat_dim]
        return m, feat_dim

    raise ValueError(f"Unknown backbone: {model_architecture}")

# ------------------------- head + wrapper -------------------------

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.head(x)


class BackboneWithHead(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(feat_dim, num_classes, dropout)

    def forward(self, x, return_feats=False):
        feats = self.backbone(x)           # [B, D]
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        logits = self.head(feats)          # [B, C]
        return (logits, feats) if return_feats else logits


def build_backbone_model(args):
    backbone, feat_dim = get_feature_backbone(args.model, pretrained=args.pretrained)
    model = BackboneWithHead(backbone=backbone, feat_dim=feat_dim, num_classes=args.num_classes, dropout=0.1)
    return model