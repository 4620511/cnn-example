import torchvision.transforms as T


def get_transforms() -> T.Compose:
    return T.Compose([T.Resize((224, 224)), T.ToTensor()])
