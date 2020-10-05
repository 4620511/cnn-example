import torchvision.transforms as T

from cnn_example.config import Config


def get_transforms(config: Config, train: bool) -> T.Compose:
    transforms = []

    if train:
        if config.preprocess.horizontal_flip:
            transforms.append(T.RandomHorizontalFlip(p=config.preprocess.horizontal_flip_rate))
        if config.preprocess.random_rotation:
            transforms.append(T.RandomRotation(degrees=config.preprocess.random_rotation_degrees))

    transforms += [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)
