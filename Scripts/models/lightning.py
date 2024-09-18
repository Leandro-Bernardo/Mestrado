import torch
import os
import json
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import pandas as pd

from random import randint

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import FloatTensor, UntypedStorage
from torch.nn import ModuleDict
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, JaccardIndex, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection, Precision, Recall, SymmetricMeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError
from typing import Any, List, Tuple, TypeVar, Optional, Dict

class CustomDataset(Dataset):
    def __init__(self, *, dataset_root: str, stage: str, dataset_idx: int = 0):
        self.dataset_root = dataset_root
        self.stage = stage
        self.dataset_idx = dataset_idx
        self.mapping = pd.read_csv(os.path.join(self.dataset_root, f"{self.stage}_mapping.csv")).set_index("descriptor_index")
        # holds one dataset in memory
        print(f"Initializing {stage} dataset")
        self.current_dataset = torch.load(os.path.join(self.dataset_root, f"descriptors_{self.stage}_dataset_{self.dataset_idx}.pt"))
        self.current_dataset_anotation = torch.load(os.path.join(self.dataset_root, f"descriptors_anotation_{self.stage}_dataset_{self.dataset_idx}.pt"))

    def __len__(self):
        with open(os.path.join(self.dataset_root, f'metadata_{self.stage}.json'), "r") as file:
            metadata = json.load(file)
            total_samples = metadata['total_samples']
            cnn1_output_size = metadata['image_size']
        return int(total_samples*cnn1_output_size)

    def __getitem__(self, idx):
        sample_dataset_num = self.mapping.loc[idx, "dataset_num"]  #the dataset (splited) that contains the required sample
        sample_dataset_idx = self.mapping.loc[idx, "dataset_descriptor_index"]  #the position of the current sample (idx) on the specific dataset
        if sample_dataset_num == self.dataset_idx:
            # gets the descriptor (X)
            descriptor_sample = self.current_dataset[sample_dataset_idx]
            # gets the anotation (Y)
            expected_value_sample = self.current_dataset_anotation[sample_dataset_idx]

        elif sample_dataset_num != self.dataset_idx:
            print(f"changed dataset from {self.dataset_idx} to {sample_dataset_num} ")
            # clears memory
            del self.current_dataset
            del self.current_dataset_anotation
            torch.cuda.empty_cache()
            # updates tracked index
            self.dataset_idx = sample_dataset_num
            # updates dataset
            self.current_dataset = torch.load(os.path.join(self.dataset_root, f"descriptors_{self.stage}_dataset_{self.dataset_idx}.pt"))
            self.current_dataset_anotation = torch.load(os.path.join(self.dataset_root, f"descriptors_anotation_{self.stage}_dataset_{self.dataset_idx}.pt"))
            # gets the descriptor (X)
            descriptor_sample = self.current_dataset[sample_dataset_idx]
            # gets the anotation (Y)
            expected_value_sample = self.current_dataset_anotation[sample_dataset_idx]

        return (descriptor_sample, expected_value_sample)

class DataModule(LightningDataModule):
    def __init__(self, *, descriptor_root: str, stage: str, batch_size: int, num_datasets_training: int = 1, num_datasets_validation: int = 1, num_datasets_test: int = 1, num_workers: int):
        super().__init__()
        self.descriptor_root = descriptor_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        #TODO validar se use_persistent_workers() funciona da forma correta
        self.num_datasets_training = num_datasets_training
        self.num_datasets_validation = num_datasets_validation
        self.num_datasets_test = num_datasets_test
        self.save_hyperparameters() # saves hyperparameters in checkpoint file

    def setup(self, stage: str) -> None:  # all the same because the dataset is splited and saved in disk
        if stage == "fit":
            self.train_subset = CustomDataset(dataset_root=self.descriptor_root, stage="train")
            self.val_subset = CustomDataset(dataset_root=self.descriptor_root, stage="val")
        elif stage == "validate":
            self.val_subset = CustomDataset(dataset_root=self.descriptor_root, stage="val")
        elif stage == "test":
            self.test_subset = CustomDataset(dataset_root=self.descriptor_root, stage="test")

    def train_dataloader(self): #NOTE disabled persistent_workers so datasets wont be saved in memory
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_subset,  batch_size=1, num_workers=self.num_workers, persistent_workers=False, shuffle=False, drop_last=True, pin_memory=True)

    def use_persistent_workers(self):
        if self.num_datasets_training > 1 or self.num_datasets_validation > 1 or self.num_datasets_test > 1:
            return False
        elif self.num_datasets_training == 1 and self.num_datasets_validation == 1 and self.num_datasets_test == 1:
            return True

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
                                                                "MAE": MeanAbsoluteError(compute_on_cpu=True),
                                                                "MAPE": MeanAbsolutePercentageError(compute_on_cpu=True),
                                                                "MSE": MeanSquaredError(compute_on_cpu=True),
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


    def on_batch_end():

        pass
    def predict_step(self, batch, batch_idx):
        # change model to evaluation mode
        #model.eval()
        # variables
        partial_loss = []
        predicted_value = []
        expected_value = []
        #total_samples = len(eval_loader)
        # disable gradient calculation
        with torch.no_grad():
            for X_batch, y_batch in batch:

                y_pred = self.model(X_batch).squeeze(1)
                predicted_value.append(round(y_pred.item(), 2))

                expected_value.append(y_batch.item())

                loss = self.criterion(y_pred, y_batch)
                partial_loss.append(loss.item())

        partial_loss = np.array(partial_loss)
        predicted_value = np.array(predicted_value)
        expected_value = np.array(expected_value)

        return partial_loss, predicted_value, expected_value # ,accuracy
