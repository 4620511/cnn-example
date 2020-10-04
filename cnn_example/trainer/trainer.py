from cnn_example.config import Config

from .resnet18 import ResNet18Module  # noqa: F401
from .resnet50 import ResNet50Module  # noqa: F401


def get_trainer(config: Config):
    if config.train.model == "resnet18":
        return ResNet18Module
    if config.train.model == "resnet50":
        return ResNet50Module
    raise ValueError("invalid model name: {}".format(config.train.model))
