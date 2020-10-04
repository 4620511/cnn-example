import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from cnn_example.config import Config
from cnn_example.data import DataModule
from cnn_example.trainer import get_trainer
from cnn_example.utils import experiment

torch.backends.cudnn.benchmark = True  # type: ignore

SEED = 42


@hydra.main(config_path="config", config_name="config")
def main(cfg: Config):
    model = get_trainer(cfg)(cfg)
    data = DataModule(cfg)
    tensorboard_logger = TensorBoardLogger(name=cfg.logger.name, save_dir=cfg.logger.save_dir)
    trainer = Trainer(
        logger=tensorboard_logger,
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.num_epochs,
        distributed_backend=None if len(cfg.train.gpus) < 2 else "ddp",
    )
    trainer.fit(model, data)  # type: ignore


if __name__ == "__main__":
    experiment.seed(SEED)
    main()
