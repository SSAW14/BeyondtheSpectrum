from .resnet import get_resnet
from .resnet_cifar import get_cifar_resnet

def get_classification_model(arch, pretrained, **kwargs):
    return get_resnet(arch, pretrained, **kwargs)

def get_cifar_classification_model(arch, pretrained, **kwargs):
    return get_cifar_resnet(arch, pretrained, **kwargs)
