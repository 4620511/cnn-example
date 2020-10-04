import os.path

import hydra
import torchvision
from pytorch_lightning import LightningDataModule

from cnn_example.config import Config
from cnn_example.data.dataloader import get_dataloader
from cnn_example.data.transforms import get_transforms


class DataModule(LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        cwd = hydra.utils.get_original_cwd()
        data_root = os.path.join(cwd, self._config.data.root)

        dataset: torchvision.datasets.VisionDataset = torchvision.datasets.__dict__[self._config.data.name.upper()]
        self._train_dataset = dataset(  # type: ignore
            root=data_root, train=True, download=True, transform=get_transforms(self._config)
        )
        self._val_dataset = dataset(  # type: ignore
            root=data_root, train=False, download=True, transform=get_transforms(self._config)
        )

    def train_dataloader(self):
        return get_dataloader(self._train_dataset, self._config.train.batch_size, self._config.train.num_workers, True)

    def val_dataloader(self):
        return get_dataloader(self._val_dataset, self._config.train.batch_size, self._config.train.num_workers, False)

    def test_dataloader(self):
        return get_dataloader(self._val_dataset, self._config.train.batch_size, self._config.train.num_workers, False)
