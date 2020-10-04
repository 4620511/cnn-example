import torchvision.transforms as T

from cnn_example.config import Config


def get_transforms(config: Config) -> T.Compose:
    transforms = []

    if config.preprocess.horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(p=config.preprocess.horizontal_flip_rate))

    transforms += [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)
