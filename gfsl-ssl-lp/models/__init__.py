from .convnet import convnet4
from .resnet import resnet12, resnet18, subresnet12
from .resnet import seresnet12
from .wresnet import wrn_28_10

# from .resnet_new import resnet50

model_pool = [
    'convnet4',
    'resnet12',
    'resnet18',
    'subresnet12',
    'seresnet12',
    'wrn_28_10',
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12': resnet12,
    'subresnet12':subresnet12,
    'resnet18': resnet18,
    'seresnet12': seresnet12
}
