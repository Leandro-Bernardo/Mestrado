import torch
import os
import json
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import FloatTensor, UntypedStorage
from torch.nn import ModuleDict
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, JaccardIndex, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection, Precision, Recall, SymmetricMeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError
from typing import Any, List, Tuple, TypeVar, Optional, Dict



class DataModule(LightningDataModule):
    def __init__(self, *, descriptor_root: str, batch_size: int , num_workers: int):
        super().__init__()
        self.descriptor_root = descriptor_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters() # saves hyperparameters in checkpoint file

    def _load_dataset(self, descriptor_root: str, stage: str):
        with open(os.path.join(descriptor_root, f'metadata_{stage}.json'), "r") as file:
            metadata = json.load(file)
        total_samples = metadata['total_samples']
        image_shape = metadata['image_shape']
        image_size = metadata['image_size']
        descriptor_depth = metadata['descriptor_depth']
        nbytes_float32 = torch.finfo(torch.float32).bits//8
        #NOTE:
        #NOTE 2: changed (now both are written and readed in same format)
        # at the moment, descriptors are saved in the format (num samples, image_size, descriptors_depth), but they are read in format (num samples * image_size,descriptors_depth).
        # expected_value is saved in format (num samples, image_size), and read in format (num samples * image_size)
        # NOTE 3 alterado shared para true
        descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(self.descriptor_root, f"descriptors_{stage}.bin"), shared = True, nbytes= (total_samples * image_size * descriptor_depth) * nbytes_float32)).view(total_samples * image_size, descriptor_depth)
        expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(self.descriptor_root, f"descriptors_anotation_{stage}.bin"), shared = True, nbytes= (total_samples * image_size) * nbytes_float32)).view(total_samples * image_size)

        #descriptors = torch.tensor(UntypedStorage.from_file(os.path.join(self.descriptor_root, f"descriptors_{stage}.bin"), shared = True, nbytes= (total_samples * image_size * descriptor_depth) * nbytes_float32), device= 'cuda').view(total_samples * image_size, descriptor_depth)
        #expected_value = torch.tensor(UntypedStorage.from_file(os.path.join(self.descriptor_root, f"descriptors_anotation_{stage}.bin"), shared = True, nbytes= (total_samples * image_size) * nbytes_float32), device= 'cuda').view(total_samples * image_size)

        return TensorDataset(descriptors, expected_value)

    def setup(self, stage: str) -> None:  # all the same because the dataset is splited and saved in disk
        if stage == "fit":
            self.train_subset = self._load_dataset(self.descriptor_root, "train")
            self.val_subset = self._load_dataset(self.descriptor_root, "val")
        elif stage == "validate":
            self.val_subset = self._load_dataset(self.descriptor_root, "val")
        elif stage == "test":
            self.test_subset = self._load_dataset(self.descriptor_root, "test")

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_subset,  batch_size=1, num_workers=self.num_workers, persistent_workers=True, shuffle=False, drop_last=True)

class BaseModel(LightningModule):
    def __init__(self, *, dataset: DataLoader, model: torch.nn.Module, batch_size: int, loss_function: torch.nn.Module, learning_rate: float, learning_rate_patience: int = None, descriptor_depth: int, sweep_config: Dict, **kwargs: Any):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.model = model(descriptor_depth, sweep_config, **kwargs)
        self.criterion = loss_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
                                                                "MAE": MeanAbsoluteError(),
                                                                "MAPE": MeanAbsolutePercentageError(),
                                                                "MSE": MeanSquaredError(),
                                                                #"WMAPE": WeightedMeanAbsolutePercentageError(),
                                                                #"SMAPE": SymmetricMeanAbsolutePercentageError(),
                                                               }) for mode_name in ["Train", "Val", "Test"]})
        #self.early_stopping_patience = early_stopping_patience

    def configure_optimizers(self):
        self.optimizer = SGD(self.parameters(), lr = self.learning_rate)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.learning_rate_patience)

        #return optimizer
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.reduce_lr_on_plateau, "monitor": "Loss/Val"}}
        #return [self.optmizer], [self.reduce_lr_on_plateau]

    # def configure_callbacks(self) -> List[Callback]:
    # # Apply early stopping.
    #  return [EarlyStopping(monitor="Loss/Val", mode="min", patience=self.early_stopping_patience)]

    def forward(self, x: Any):
        return self.model(x)

    # TODO verificar o dado de entrada
    #defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, torch.tensor], stage: str):
        X, y = batch[0].squeeze(), batch[1].squeeze()
        predicted_value = self(X)    # o proprio objeto de BaseModel é o modelo (https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
        predicted_value = predicted_value.squeeze()
        # Compute and log the loss value.
        loss = self.criterion(predicted_value, y)
        self.log(f"Loss/{stage}", loss, prog_bar=True)
        # Compute and log step metrics.
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Step': value for metric_name, value in metrics(predicted_value, y).items()})
        return loss

    def training_step(self, batch: List[torch.tensor]):#, batch_idx: int):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: List[torch.tensor]):#, batch_idx: int):
        return self._any_step(batch, "Val")

    def test_step(self, batch: List[torch.tensor]):#, batch_idx: int):
        return self._any_step(batch, "Test")

    def _any_epoch_end(self, stage: str):
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Epoch': value for metric_name, value in metrics.compute().items()}, on_step=False, on_epoch=True) # logs metrics on epoch end
        metrics.reset()
        # Print loss at the end of each epoch
        #loss = self.trainer.callback_metrics[f"Loss/{stage}"]
        #print(f"Epoch {self.current_epoch} - Loss/{stage}: {loss.item()}")

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")


    # def predict_step(self, batch, batch_idx):
    #     # change model to evaluation mode
    #     #model.eval()
    #     # variables
    #     partial_loss = []
    #     predicted_value = []
    #     expected_value = []
    #     #total_samples = len(eval_loader)
    #     # disable gradient calculation
    #     with torch.no_grad():
    #         for X_batch, y_batch in batch:

    #             y_pred = self.model(X_batch).squeeze(1)
    #             predicted_value.append(round(y_pred.item(), 2))

    #             expected_value.append(y_batch.item())

    #             loss = self.criterion(y_pred, y_batch)
    #             partial_loss.append(loss.item())

    #     partial_loss = np.array(partial_loss)
    #     predicted_value = np.array(predicted_value)
    #     expected_value = np.array(expected_value)

    #     return partial_loss, predicted_value, expected_value # ,accuracy
