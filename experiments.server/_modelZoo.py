# e2eET Skeleton Based HGR Using Data-Level Fusion
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportWildcardImportFromLibrary=false
# -----------------------------------------------
from fastai.vision.all import *
import pytorchcv.model_provider as pcvm
import geffnet as genm

__all__ = [
    "BaseArchitectures", 
]
# -----------------------------------------------


cache = "~/.cache/torch/hub/pytorchcv"
gen_params = {"drop_connect_rate": 0.2, "as_sequential": True}

# ---
def inception_v3(pretrained=True): 
    return pcvm.inceptionv3(pretrained=pretrained, root=cache).features
def inception_v4(pretrained=True): 
    return pcvm.inceptionv4(pretrained=pretrained, root=cache).features
def inceptionresnet_v1(pretrained=True): 
    return pcvm.inceptionresnetv1(pretrained=pretrained, root=cache).features
def inceptionresnet_v2(pretrained=True): 
    return pcvm.inceptionresnetv2(pretrained=pretrained, root=cache).features

# ---
def se_resnet18(pretrained=True): 
    return pcvm.seresnet18(pretrained=pretrained, root=cache).features
def se_resnet26(pretrained=True): 
    return pcvm.seresnet26(pretrained=pretrained, root=cache).features
def se_resnet50(pretrained=True): 
    return pcvm.seresnet50(pretrained=pretrained, root=cache).features
def se_resnet101(pretrained=True): 
    return pcvm.seresnet101(pretrained=pretrained, root=cache).features
def se_resnet152(pretrained=True): 
    return pcvm.seresnet152(pretrained=pretrained, root=cache).features

# ---
def resnext26(pretrained=True): 
    return pcvm.resnext26_32x4d(pretrained=pretrained, root=cache).features
def resnext50(pretrained=True): 
    return pcvm.resnext50_32x4d(pretrained=pretrained, root=cache).features
def resnext101(pretrained=True): 
    return pcvm.resnext101_32x4d(pretrained=pretrained, root=cache).features

# ---
def se_resnext50(pretrained=True): 
    return pcvm.seresnext50_32x4d(pretrained=pretrained, root=cache).features
def se_resnext101(pretrained=True): 
    return pcvm.seresnext101_32x4d(pretrained=pretrained, root=cache).features

# ---
def efficientnet_b0(pretrained=True):
    return genm.tf_efficientnet_b0_ns(pretrained=pretrained, drop_rate=0.2, **gen_params)
def efficientnet_b3(pretrained=True):
    return genm.tf_efficientnet_b3_ns(pretrained=pretrained, drop_rate=0.3, **gen_params)
def efficientnet_b5(pretrained=True):
    return genm.tf_efficientnet_b5_ns(pretrained=pretrained, drop_rate=0.4, **gen_params)
def efficientnet_b7(pretrained=True):
    return genm.tf_efficientnet_b7_ns(pretrained=pretrained, drop_rate=0.5, **gen_params)

# ---
BaseArchitectures = {
    # --- ResNet
    "resnet18":  models.resnet18,
    "resnet34":  models.resnet34,
    "resnet50":  models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    # --- XResNet
    "xresnet50":        models.xresnet.xresnet50,
    "xresnet50_deep":   models.xresnet.xresnet50_deep,
    "xresnet50_deeper": models.xresnet.xresnet50_deeper,
    # --- Inception
    "inception_v3":       inception_v3,
    "inception_v4":       inception_v4,
    "inceptionresnet_v1": inceptionresnet_v1,
    "inceptionresnet_v2": inceptionresnet_v2,
    # --- SE-ResNet
    "se_resnet18":  se_resnet18,
    "se_resnet26":  se_resnet26,
    "se_resnet50":  se_resnet50,
    "se_resnet101": se_resnet101,
    "se_resnet152": se_resnet152,
    # --- ResNeXt
    "resnext26":  resnext26,
    "resnext50":  resnext50,
    "resnext101": resnext101,
    # --- SE-ResNeXt
    "se_resnext50":  se_resnext50,
    "se_resnext101": se_resnext101,
    # --- EfficientNet
    "efficientnet_b0": efficientnet_b0,  # geffnet
    "efficientnet_b3": efficientnet_b3,  # geffnet
    "efficientnet_b5": efficientnet_b5,  # geffnet
    "efficientnet_b7": efficientnet_b7,  # geffnet
}
