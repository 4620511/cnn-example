import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as M
from pytorch_lightning import EvalResult, LightningModule, TrainResult

from cnn_example.config import Config


class ResNet18Module(LightningModule):
    def __init__(self, hparams: Config):
        super().__init__()
        self.hparams: Config = hparams

        self.__build_model()
        self.__build_criterion()

    def __build_model(self):
        self._model = M.resnet18(pretrained=self.hparams.train.pre_trained)
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.fc = nn.Linear(512, self.hparams.data.num_classes)

    def __build_criterion(self):
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        loss = self._criterion(outputs, targets)
        acc = torch.sum((targets == torch.argmax(outputs, dim=1))).float() / len(targets)
        result = TrainResult(loss)
        result.log("train/loss", loss, on_step=False, on_epoch=True)
        result.log("train/acc", acc, on_step=False, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        loss = self._criterion(outputs, targets)
        acc = torch.sum((targets == torch.argmax(outputs, dim=1))).float() / len(targets)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        result = EvalResult(checkpoint_on=loss)
        result.log("val/loss", loss, on_step=False, on_epoch=True)
        result.log("val/acc", acc, on_step=False, on_epoch=True)
        return result

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
