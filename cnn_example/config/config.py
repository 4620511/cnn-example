from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig


@dataclass
class Data:
    name: str
    num_classes: int
    root: str


@dataclass
class Logger:
    name: str
    save_dir: str


@dataclass
class Preprocess:
    horizontal_flip: bool
    horizontal_flip_rate: float
    random_rotation: bool
    random_rotation_degrees: int


@dataclass
class Train:
    batch_size: int
    gpus: List[int]
    model: str
    num_epochs: int
    num_workers: int
    pre_trained: bool
    tags: List[str]
    train_rate: float


@dataclass
class Config(DictConfig):
    data: Data
    logger: Logger
    preprocess: Preprocess
    train: Train
