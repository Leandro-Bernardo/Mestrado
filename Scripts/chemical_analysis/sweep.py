import pandas as pd
import torch
import numpy as np
import json, multiprocessing, os, shutil, time
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl

from plotly.subplots import make_subplots
from ._dataset import ExpandedSampleDataset, SampleDataset, ProcessedSampleDataset
from ._model import Network
from ._utils import whitebalance
from .typing import CalibratedDistributions, Loss, Values
from _const import WandbMode
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum, auto
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from tempfile import TemporaryDirectory
from torch import FloatTensor, UntypedStorage, Tensor
from torch.nn import CrossEntropyLoss, ModuleDict, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics import Accuracy, F1Score, JaccardIndex, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection, Precision, Recall, SymmetricMeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from dist2dist._utils import _check_if_artificial_pmfs_exists, _load_artificial_pmfs_from_cache


os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".
from wandb.wandb_run import Run
import wandb


class DataCheckpointAction(Enum):
    CREATE = auto()
    USE = auto()


@contextmanager
def data_checkpoint(root_dir: str, label: str):
    ok_path = os.path.join(root_dir, f'__{label}-ok')
    try:
        if os.path.exists(ok_path):
            print(f'Using "{label}" data from "{root_dir}"', flush=True)
            yield DataCheckpointAction.USE
        else:
            creating_path = os.path.join(root_dir, f'__{label}-creating')
            if os.path.exists(creating_path):
                print(f'Waiting for another process to create "{label}" data in "{root_dir}"', flush=True)
                while not os.path.exists(ok_path):
                    time.sleep(1)
                print(f'Using "{label}" data from "{root_dir}"', flush=True)
                yield DataCheckpointAction.USE
            else:
                try:
                    with open(creating_path, "a"):
                        print(f'Creating "{label}" data in "{root_dir}"', flush=True)
                        yield DataCheckpointAction.CREATE
                finally:
                    if os.path.exists(creating_path):
                        os.remove(creating_path)
    finally:
        pass
    with open(ok_path, "a"):
        pass


class DataModule(LightningDataModule):
    def __init__(self, *, batch_size: int, dataset_root_dir: str, fit_samples_base_dirs: Iterable[str], num_augmented_samples: int, sample_dataset_class: Type[SampleDataset], processed_sample_dataset_class: Type[ProcessedSampleDataset], reduction_level: float, test_samples_base_dirs: Iterable[str], use_expanded_set: bool, use_artificial_data: bool, val_proportion: float, **_: Any) -> None:
        super().__init__()
        # Keep the input arguments.
        self.batch_size = batch_size
        self.dataset_root_dir = dataset_root_dir
        self.fit_samples_base_dirs = fit_samples_base_dirs
        self.num_augmented_samples = num_augmented_samples
        self.processed_sample_dataset_class = processed_sample_dataset_class
        self.reduction_level = reduction_level
        self.sample_dataset_class = sample_dataset_class
        self.test_samples_base_dirs = test_samples_base_dirs
        self.use_expanded_set = use_expanded_set
        self.val_proportion = val_proportion
        self.use_artificial_data = use_artificial_data
        # Set other arguments.
        self.artifact_name = f'{sample_dataset_class.__name__}-{"Expanded" if use_expanded_set else "NotExpanded"}-AugmentedSamples_{num_augmented_samples}-ReductionLevel_{reduction_level:1.2f}-ValProportion_{val_proportion:1.2f}'
        self.pca_stats_filepath = os.path.join(dataset_root_dir, "PcaStats.npz")

    def _compute_pca_stats(self, subset: ProcessedSampleDataset) -> Dict[str, np.ndarray]:
        labs = np.empty((3, len(subset)), dtype=np.float32)
        for index, item in enumerate(tqdm(iter(subset), total=len(subset), desc="Computing PCA statistics", leave=False)):
            analyte_msk = item.sample_analyte_mask
            lab = whitebalance(item.sample_lab_image[analyte_msk], item.sample_lab_white)
            labs[:, index] = lab.sum(axis=0)
        mean = labs.mean(axis=1, keepdims=True)
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(labs - mean, rowvar=True))
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        return {"lab_mean": mean.squeeze(axis=1), "lab_sorted_eigenvalues": sorted_eigenvalues, "lab_sorted_eigenvectors": sorted_eigenvectors}

    def _load_ready_to_use_subset(self, split: str, use_artificial_data:bool=False) -> TensorDataset:
        # Loading artificial PMFs if needed
        if split == "train" and use_artificial_data: # TODO adaptar agora para pegar apenas dados artificiais
            if not _check_if_artificial_pmfs_exists(stage=split, dataset_root_dir=self.dataset_root_dir):
                raise Exception("There is no artificial PMFs in cache.")
            augmented_pmfs, augmented_expected_values = _load_artificial_pmfs_from_cache(stage=split, dataset_root_dir=self.dataset_root_dir)
            return TensorDataset(augmented_pmfs, augmented_expected_values)
        # Loading real PMFs
        with open(os.path.join(self.dataset_root_dir, f'{split}-processed_samples.json'), "r") as fin:
            header = json.load(fin)
        nbytes_float32 = torch.finfo(torch.float32).bits // 8
        num_samples = header["num_samples"]
        calibrated_pmf_shape = header["calibrated_pmf_shape"]
        calibrated_pmf = FloatTensor(UntypedStorage.from_file(os.path.join(self.dataset_root_dir, f'{split}-calibrated_pmf.bin'), shared=True, nbytes=int((np.prod(calibrated_pmf_shape).item() * nbytes_float32 * num_samples)))).view(num_samples, *calibrated_pmf_shape)
        expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(self.dataset_root_dir, f'{split}-expected_value.bin'), shared=True, nbytes=(nbytes_float32 * num_samples)))
        return TensorDataset(calibrated_pmf, expected_value)

    def _write_ready_to_use_subset(self, split: str, subset: ProcessedSampleDataset) -> None:
        num_samples = len(subset)
        if num_samples > 0:
            nbytes_float32 = torch.finfo(torch.float32).bits // 8
            calibrated_pmf_shape = tuple(subset[0].calibrated_pmf.shape)
            calibrated_pmf = FloatTensor(UntypedStorage.from_file(os.path.join(self.dataset_root_dir, f'{split}-calibrated_pmf.bin'), shared=True, nbytes=int((np.prod(calibrated_pmf_shape).item() * nbytes_float32 * num_samples)))).view(num_samples, *calibrated_pmf_shape)
            expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(self.dataset_root_dir, f'{split}-expected_value.bin'), shared=True, nbytes=(nbytes_float32 * num_samples)))
            for index, item in enumerate(tqdm(iter(subset), total=len(subset), desc=f'Writing "{split}" split to disk', leave=False)):
                calibrated_pmf[index, ...] = torch.as_tensor(item.calibrated_pmf, dtype=torch.float32)
                assert item.sample["correctedTheoreticalValue"] is not None
                expected_value[index] = item.sample["correctedTheoreticalValue"]
        else:
            calibrated_pmf_shape = tuple()
            calibrated_pmf = FloatTensor(UntypedStorage.from_file(os.path.join(self.dataset_root_dir, f'{split}-calibrated_pmf.bin'), shared=True, nbytes=0))
            expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(self.dataset_root_dir, f'{split}-expected_value.bin'), shared=True, nbytes=0))
        with open(os.path.join(self.dataset_root_dir, f'{split}-processed_samples.json'), "w") as fout:
            json.dump({"num_samples": num_samples, "calibrated_pmf_shape": calibrated_pmf_shape}, fout)

    def data_parameters(self) -> Dict[str, Any]:
        return dict(np.load(os.path.join(self.dataset_root_dir, "DataParameters.npz")))

    def prepare_data(self) -> None:
        create_artifact = False
        os.makedirs(self.dataset_root_dir, exist_ok=True)
        # Check whether the ready-to-use data is available.
        with data_checkpoint(self.dataset_root_dir, "ready_to_use") as action:
            # If data is not available them try to download it or create it.
            if action == DataCheckpointAction.CREATE:
                try:
                    # Try to download the artifact from W&B and load the stored subset from downloaded data.
                    artifact = wandb.use_artifact(f'{self.artifact_name}:latest', type="dataset")
                    artifact.wait()
                    artifact.download(self.dataset_root_dir)
                    artifact.wait()
                except:
                    # If the artifact does not exist them make the train, validation, and test subsets of processed samples...
                    fit_dataset = self.sample_dataset_class(self.fit_samples_base_dirs, skip_blank_samples=True, skip_incomplete_samples=True, skip_inference_sample=True, skip_training_sample=False)
                    num_val = int(len(fit_dataset) * self.val_proportion)
                    num_train = len(fit_dataset) - num_val
                    train_subset, val_subset = random_split(fit_dataset, [num_train, num_val])
                    if self.use_expanded_set:
                        train_subset = ExpandedSampleDataset(train_subset)
                    test_subset = self.sample_dataset_class(self.test_samples_base_dirs, skip_blank_samples=True, skip_incomplete_samples=True, skip_inference_sample=True, skip_training_sample=False)
                    # ... compute and write PCA statistics, ...
                    with TemporaryDirectory(dir=self.dataset_root_dir, prefix=".cache-pca_stats-") as tmpdir:
                        processed_subset = self.processed_sample_dataset_class(train_subset, cache_dir=tmpdir, num_augmented_samples=self.num_augmented_samples, lab_mean=np.zeros((3,), dtype=np.float32), lab_sorted_eigenvectors=np.eye(3, dtype=np.float32))
                        pca_stats = self._compute_pca_stats(processed_subset)
                        np.savez_compressed(self.pca_stats_filepath, **pca_stats)
                    # ... and compute and write processed data as stored tensors.
                    with TemporaryDirectory(dir=self.dataset_root_dir, prefix=".cache-train-") as tmpdir:
                        processed_subset = self.processed_sample_dataset_class(train_subset, cache_dir=tmpdir, num_augmented_samples=self.num_augmented_samples, **pca_stats)
                        self._write_ready_to_use_subset("train", processed_subset)
                        training_stats = processed_subset.compute_true_value_statistics()
                        input_roi, input_range = processed_subset.compute_calibrated_pmf_roi(self.reduction_level)
                        np.savez_compressed(os.path.join(self.dataset_root_dir, "DataParameters.npz"), input_range=input_range, input_roi=input_roi, training_mad=training_stats["mad"], training_median=training_stats["median"])
                        with open(os.path.join(self.dataset_root_dir, "train-samples.json"), "w") as fout:
                            json.dump({
                                "use_expanded_set": self.use_expanded_set,
                                "num_augmented_samples": self.num_augmented_samples,
                                "original_samples": list(sorted([os.path.splitext(os.path.basename(item["fileName"]))[0] for item in train_subset])),
                            }, fout)
                    with TemporaryDirectory(dir=self.dataset_root_dir, prefix=".cache-val-") as tmpdir:
                        processed_subset = self.processed_sample_dataset_class(val_subset, cache_dir=tmpdir, num_augmented_samples=0, **pca_stats)
                        self._write_ready_to_use_subset("val", processed_subset)
                        with open(os.path.join(self.dataset_root_dir, "val-samples.json"), "w") as fout:
                            json.dump({
                                "use_expanded_set": False,
                                "num_augmented_samples": 0,
                                "original_samples": list(sorted([os.path.splitext(os.path.basename(item["fileName"]))[0] for item in val_subset])),
                            }, fout)
                    with TemporaryDirectory(dir=self.dataset_root_dir, prefix=".cache-test-") as tmpdir:
                        processed_subset = self.processed_sample_dataset_class(test_subset, cache_dir=tmpdir, num_augmented_samples=0, **pca_stats)
                        self._write_ready_to_use_subset("test", processed_subset)
                        with open(os.path.join(self.dataset_root_dir, "test-samples.json"), "w") as fout:
                            json.dump({
                                "use_expanded_set": False,
                                "num_augmented_samples": 0,
                                "original_samples": list(sorted([os.path.splitext(os.path.basename(item["fileName"]))[0] for item in test_subset])),
                            }, fout)
                    # We have to schedule to upload the artfact.
                    create_artifact = True
            # If data is available them do nothing.
            elif action == DataCheckpointAction.USE:
                pass  # Do nothing.
            # If action is unknown them we have a bug!
            else:
                raise NotImplementedError
        # Create and upload the artifact, if needed.
        if create_artifact:
            artifact = wandb.Artifact(self.artifact_name, type="dataset")
            artifact.add_dir(self.dataset_root_dir)
            wandb.log_artifact(artifact)
            artifact.wait()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_subset = self._load_ready_to_use_subset("train", use_artificial_data=self.use_artificial_data)
            self.val_subset = self._load_ready_to_use_subset("val")
        elif stage == "validate":
            self.val_subset = self._load_ready_to_use_subset("val")
        elif stage == "test":
            self.test_subset = self._load_ready_to_use_subset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=min(2, multiprocessing.cpu_count()), shuffle=True, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=min(2, multiprocessing.cpu_count()), shuffle=False, pin_memory=True, drop_last=False, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_subset, batch_size=self.batch_size, num_workers=min(2, multiprocessing.cpu_count()), shuffle=False, pin_memory=True, drop_last=False, persistent_workers=True)

    @classmethod
    def wandb_parameters(cls) -> Dict[str, Any]:
        return {"batch_size": {"distribution": "int_uniform", "min": 5, "max": 30}}


class BaseModel(ABC, LightningModule):
    def __init__(self, *, early_stopping_patience: int, learning_rate: float, learning_rate_patience: int, network_class: Type[Network], **kwargs: Any) -> None:
        super().__init__()
        # Keep the input arguments.
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.net = network_class(**kwargs)
        self.inf_vs_target = {"Train": list(), "Val": list(), "Test": list()}
        self._error_zones: List[float] = [2.5, 5, 10, 20, 100]

    @abstractmethod
    def _any_epoch_end(self, mode_name: str) -> None:
        raise NotImplementedError  # To be implemented by the subclass.

    @abstractmethod
    def _any_step(self, batch: Tuple[CalibratedDistributions, Values], mode_name: str) -> Loss:
        raise NotImplementedError  # To be implemented by the subclass.

    def configure_callbacks(self) -> List[Callback]:
        # Apply early stopping.
        return [EarlyStopping(monitor="Loss/Val", mode="min", patience=self.early_stopping_patience)]

    def configure_optimizers(self) -> Dict[str, Any]:
        # Set the optimizer.
        optimizer = SGD(self.net.parameters(), momentum=0.9, lr=self.learning_rate)
        # Set the learning rate scheduler.
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=self.learning_rate_patience)
        # Return the configuration.
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "Loss/Val"}}

    def forward(self, input: Any) -> Any:
        return self.net(input)

    def on_test_epoch_end(self) -> None:
        self._any_epoch_end("Test")

    def on_train_epoch_end(self) -> None:
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self) -> None:
        self._any_epoch_end("Val")

    def training_step(self, batch: Tuple[CalibratedDistributions, Values], batch_idx: int) -> Loss:
        return self._any_step(batch, "Train")

    def validation_step(self, batch: Tuple[CalibratedDistributions, Values], batch_idx: int) -> None:
        self._any_step(batch, "Val")

    def test_step(self, batch: Tuple[CalibratedDistributions, Values], batch_idx: int) -> None:
        self._any_step(batch, "Test")
        
    def on_train_end(self) -> None:
        self.log_last_inferences(mode="Train")
        self.log_last_inferences(mode="Val")
    
    def log_last_inferences(self, mode):
        lower_interval, upper_interval = self.net.expected_range
        predicted_values = self.inf_vs_target[mode][0]["predicted"]
        target_values = self.inf_vs_target[mode][0]["target"]
        data_size = len(target_values)
        # Computing statistics
        absolute_errors = torch.abs(predicted_values - target_values)
        relative_errors = absolute_errors/torch.max(target_values, torch.as_tensor(torch.finfo(torch.float32).eps))
        # Testar qual dos dois funciona para substituir o max
        percentage_error = relative_errors * 100
        # Computing relative percent error in intervals
        relative_error_in_range = list(map(lambda x, y: (x <= y), [percentage_error]*len(self._error_zones), self._error_zones))
        qtt_samples_in_interval, interval_labels = self.format_data_for_pie_chart(relative_error=relative_error_in_range, data_size=data_size)
        target_and_error_in_expected_range = [(x[0], x[1]) for x in zip(target_values, percentage_error) if x[0]>=lower_interval and x[0]<=upper_interval]
        relative_error_in_expected_range = torch.tensor(list(map(lambda x: x[1], target_and_error_in_expected_range)))
        relative_error_in_range = list(map(lambda x, y: (x <= y), [relative_error_in_expected_range]*len(self._error_zones), self._error_zones))
        qtt_expected_samples_in_intervals, _ = self.format_data_for_pie_chart(relative_error=relative_error_in_range, data_size=len(target_and_error_in_expected_range))
        # Sortting data
        sorted_target, indxs = list(map(lambda x: x.tolist(), torch.sort(target_values)))
        sorted_prediction = [pred[i] for pred, i in zip([predicted_values.tolist()]*len(indxs), indxs)]
        sorted_ae = [m[i] for m, i in zip([absolute_errors.tolist()]*len(indxs), indxs)]
        sorted_re = [m[i] for m, i in zip([relative_errors.tolist()]*len(indxs), indxs)]
        # Converting statistics into DataFrame
        df = pd.DataFrame({
            'Target': sorted_target,
            'Prediction': sorted_prediction,
            'Relative Error': sorted_re,
            'Absolute Error': sorted_ae,
            'Index': [i for i in range(len(sorted_target))],
        })
        df_expected_range = pd.DataFrame({
            'Target': [target for target in sorted_target if target>=lower_interval and target<=upper_interval],
            'Prediction': [prediction for target, prediction in zip(sorted_target, sorted_prediction) if target>=lower_interval and target<=upper_interval],
            'Relative Error': [error for target, error in zip(sorted_target, sorted_re) if target>=lower_interval and target<=upper_interval],
            'Absolute Error': [error for target, error in zip(sorted_target, sorted_ae) if target>=lower_interval and target<=upper_interval],
            'Index': [x for x in range(len([target for target in sorted_target if target>=lower_interval and target<=upper_interval]))]
        })
        # Gatting information to configure plot axis range
        target_min, target_max = sorted_target[0], sorted_target[-1:][0]
        offset = target_max*0.1
        # Plotting the target values in contrast with the predicted ones
        fig = px.scatter(df, x="Target", y="Prediction", color='Relative Error', hover_data=['Target', 'Prediction', 'Relative Error', 'Absolute Error'], height=400, width=400, trendline="ols", color_continuous_scale=px.colors.sequential.Agsunset)
        fig.add_scatter(x=sorted_target, y=sorted_target, mode='lines', showlegend=False)
        in_range = [(target, prediction) for target, prediction in zip(sorted_target, sorted_prediction) if target>=lower_interval and target<=upper_interval]
        target_in_range = [i[0] for i in in_range]
        prediction_in_range = [i[1] for i in in_range]
        trend = np.polyfit(target_in_range, prediction_in_range, 1)
        fig.add_scatter(x=sorted_target, y=[trend[0]*i + trend[1] for i in sorted_target], mode='lines', showlegend=False)
        fig.update_layout(yaxis_range=[target_min-offset, target_max+offset], xaxis_range=[target_min-offset, target_max+offset])
        wandb.log({f"Predictions/{mode}": fig})
        # Plotting the absolute error
        fig = px.bar(df, x='Index', y='Absolute Error', hover_data=['Target', 'Prediction', 'Relative Error', 'Absolute Error'],
                     color='Target', labels={'Target':'ppm'}, height=400, color_continuous_scale=px.colors.sequential.Agsunset)
        wandb.log({f"Absolute Error/Over All ({mode})": fig})
        fig = px.bar(df_expected_range, x='Index', y='Absolute Error', hover_data=['Target', 'Prediction', 'Relative Error', 'Absolute Error'],
                     color='Target', labels={'Target':'ppm'}, height=400, color_continuous_scale=px.colors.sequential.Agsunset)
        wandb.log({f"Absolute Error/Expected Range ({mode})": fig})
        # Plotting the relative percent errors
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'polar'}, {'type':'polar'}]], subplot_titles=['Over All', 'Expected Range'])
        fig.add_trace(go.Barpolar(theta=interval_labels, r=qtt_samples_in_interval, marker_color=px.colors.sequential.Agsunset, showlegend=False), row=1, col=1)
        fig.add_trace(go.Barpolar(theta=interval_labels, r=qtt_expected_samples_in_intervals, marker_color=px.colors.sequential.Agsunset, showlegend=False), row=1, col=2)
        # fig.update_traces(marker_color=px.colors.sequential.Agsunset)
        wandb.log({f"Relative Percentage Error in Interval/{mode}": fig})
        
    def format_data_for_pie_chart(self, relative_error, data_size: int) -> Tuple[List[Tensor], List[str]]:
        qtt_samples_in_interval = list(map(lambda x: torch.sum(x).item(), relative_error))
        qtt_samples_in_interval = [qtt_samples_in_interval[0]]+list(map(lambda x, y: x-y, qtt_samples_in_interval[1:], qtt_samples_in_interval[:-1]))
        qtt_samples_in_interval.append(data_size-sum(qtt_samples_in_interval))
        interval_labels = list(map(lambda x: f"[{x[0]}%, {x[1]}%)", zip([0]+self._error_zones, self._error_zones)))
        interval_labels.append(f"[{self._error_zones[-1:][0]}%, inf)")
        return qtt_samples_in_interval, interval_labels

    @classmethod
    @abstractmethod
    def wandb_metric(cls) -> Dict[str, Any]:
        raise NotImplementedError  # To be implemented by the subclass.

    @classmethod
    def wandb_parameters(cls) -> Dict[str, Any]:
        return {"learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-1}}


class ContinuousModel(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Save hyper-parameters to self.hparams. They will also be automatically logged as config parameters in Weights & Biases.
        self.save_hyperparameters()
        # Set the loss function.
        self.criterion = MSELoss()
        # Set metrics.
        self.metrics = ModuleDict({mode_name: MetricCollection({
            "MAE": MeanAbsoluteError(),
            "MAPE": MeanAbsolutePercentageError(),
            "MSE": MeanSquaredError(),
            "WMAPE": WeightedMeanAbsolutePercentageError(),
            "SMAPE": SymmetricMeanAbsolutePercentageError(),
        }) for mode_name in ["Train", "Val", "Test"]})
        # Set inference vs target data for chart plot
        self.outputs = {"Train": list(), "Val": list(), "Test": list()}

    def _any_epoch_end(self, mode_name: str) -> None:
        metrics: MetricCollection = self.metrics[mode_name]  # type: ignore
        self.log_dict({f'{metric_name}/{mode_name}/Epoch': value for metric_name, value in metrics.compute().items()})
        metrics.reset()
        # Allways keep the last predicted and target values.
        self.inf_vs_target[mode_name].clear()
        self.inf_vs_target[mode_name].append({
            "predicted": torch.cat([x["predicted_value"] for x in self.outputs[mode_name]]).flatten(),
            "target": torch.cat([x["target_value"] for x in self.outputs[mode_name]]).flatten(),
        })
        # Cleans outputs before next epoch
        self.outputs[mode_name].clear()

    def _any_step(self, batch: Tuple[CalibratedDistributions, Values], mode_name: str) -> Loss:
        # Evaluate the model on the given batch.
        calibrated_pmf, target_value = batch
        predicted_value, predicted_normalized_value = self(calibrated_pmf)
        # Compute and log the loss value.
        expected_normalized_value = ((target_value - self.net.training_median) / (2.5758 * 1.4826 * self.net.training_mad) + 1.0) * 0.5
        loss = self.criterion(predicted_normalized_value, expected_normalized_value)
        self.log(f"Loss/{mode_name}", loss)
        # Compute and log step metrics.
        metrics: MetricCollection = self.metrics[mode_name]  # type: ignore
        self.log_dict({f'{metric_name}/{mode_name}/Step': value for metric_name, value in metrics(predicted_value, target_value).items()})
        # Store target and predicted values for chart plot
        with torch.no_grad():
            self.outputs[mode_name].append({
                "predicted_value": predicted_value.detach(),
                "target_value": target_value,
            })
        # Return the loss value.
        return loss

    @classmethod
    def wandb_metric(cls) -> Dict[str, Any]:
        return {"name": "Loss/Val", "goal": "minimize"}


class IntervalModel(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Save hyper-parameters to self.hparams. They will also be automatically logged as config parameters in Weights & Biases.
        self.save_hyperparameters()
        # Set the loss function.
        self.criterion = CrossEntropyLoss()
        # Set metrics.
        num_intervals = len(self.net.intervals)
        self.metrics = ModuleDict({mode_name: MetricCollection({
            "Accuracy": Accuracy(task="multiclass", num_classes=num_intervals),
            "F1-Score": F1Score(task="multiclass", num_classes=num_intervals),
            "IoU": JaccardIndex(task="multiclass", num_classes=num_intervals),
            "Precision": Precision(task="multiclass", num_classes=num_intervals),
            "Recall": Recall(task="multiclass", num_classes=num_intervals),
        }) for mode_name in ["Train", "Val", "Test"]})

    def _any_epoch_end(self, mode_name: str) -> None:
        metrics: MetricCollection = self.metrics[mode_name]  # type: ignore
        self.log_dict({f'{metric_name}/{mode_name}/Epoch': value for metric_name, value in metrics.compute().items()})
        metrics.reset()

    def _any_step(self, batch: Tuple[CalibratedDistributions, Values], mode_name: str) -> Loss:
        # Evaluate the model on the given batch.
        calibrated_pmf, target_value = batch
        target_value = target_value.unsqueeze(-1)
        intervals = self.net.intervals.to(target_value)
        logits = self(calibrated_pmf)
        target = torch.argmax(torch.logical_and(intervals[:, 0] <= target_value, target_value < intervals[:, 1]).view(dtype=torch.uint8), dim=-1)  # type: ignore
        # Compute and log the loss value.
        loss = self.criterion(logits, target)
        self.log(f"Loss/{mode_name}", loss)
        # Compute and log step metrics.
        metrics: MetricCollection = self.metrics[mode_name]  # type: ignore
        self.log_dict({f'{metric_name}/{mode_name}/Step': value for metric_name, value in metrics(logits, target).items()})
        # Return the loss value.
        return loss

    @classmethod
    def wandb_metric(cls) -> Dict[str, Any]:
        return {"name": "Accuracy/Val", "goal": "maximize"}

    @classmethod
    def wandb_parameters(cls) -> Dict[str, Any]:
        return {"num_divisions": {"values": [2, 3, 4]}, **BaseModel.wandb_parameters()}


def _tracked_run(*, checkpoint_dir: str, gpus: int, model_class: Type[BaseModel], seed: Optional[int], accelerator:str ='gpu', wandb_mode:str=WandbMode.ONLINE, **kwargs: Any) -> None:
    # Start a new tracked run at Weights & Biases.
    with wandb.init(mode=wandb_mode) as run:
        assert isinstance(run, Run)
        # Ensure full reproducibility.
        if seed is not None:
            pl.seed_everything(seed, workers=True)
        # Setup the data module.
        datamodule = DataModule(**run.config.as_dict(), **kwargs)
        datamodule.prepare_data()
        # Setup the model.
        model = model_class(**run.config.as_dict(), **datamodule.data_parameters(), **kwargs)
        # Setup the trainer.
        trainer = Trainer(
            logger=WandbLogger(experiment=run),
            accelerator=accelerator,
            devices=gpus,
            default_root_dir=checkpoint_dir,
            log_every_n_steps=10,
            max_epochs=-1,
            num_sanity_val_steps=0,
            callbacks=[
                LearningRateMonitor(logging_interval='epoch')
            ])
        # Perform fitting.
        trainer.fit(model, datamodule=datamodule)
        # Perform test.
        trainer.test(model, datamodule=datamodule)
        # Save trained model.
        checkpoint_filepath = os.path.join(checkpoint_dir, f'{kwargs["network_class"].__name__}.ckpt')
        trainer.save_checkpoint(checkpoint_filepath, weights_only=True)
        # wandb.save(checkpoint_filepath)
        # Save PCA stats.
        pca_stats_copy_filepath = os.path.join(checkpoint_dir, os.path.basename(datamodule.pca_stats_filepath))
        shutil.copy(datamodule.pca_stats_filepath, pca_stats_copy_filepath)
        wandb.save(pca_stats_copy_filepath)


def make_base_config(sweep_name: str, program: str) -> Dict[str, Any]:
    return {
        "name": sweep_name,
        "program": program,
        "method": "bayes",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 20,
        },
        "metric": {},
        "parameters": {},
    }


def resume(sweep_id: str, entity_name: str, project_name: str, **kwargs: Any) -> None:
    wandb.agent(sweep_id, function=lambda: _tracked_run(**kwargs), entity=entity_name, project=project_name)


def start(config: Dict[str, Any], entity_name: str, project_name: str, **kwargs: Any) -> None:
    sweep_id = wandb.sweep(config, entity=entity_name, project=project_name)
    wandb.agent(sweep_id, function=lambda: _tracked_run(**kwargs))
