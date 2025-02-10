import random
import numpy as np
import pytorch_lightning as pl
import torch
import itertools
from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import FloatTensor, UntypedStorage, IntTensor, Tensor
from torch.nn import ModuleDict
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MetricCollection
from tqdm import tqdm
import json, multiprocessing, os

from dist2dist.emdloss import EMDLoss
from dist2dist._utils import rotate_and_translate_in_batch, calculate_interpolation_factor, compute_affine_parameters, get_featured_roi_size, _check_if_artificial_pmfs_exists
from dist2dist.random_dataset import ItemwiseDataset
from dist2dist.equidistant_dataset import GroupwiseDataset
from dist2dist.generated_dataset import GeneratedDataset
from dist2dist.cyclic import CyclicDataset
from _const import ModelMode, DevMode, WandbMode, LightningStage, Dist2DistCacheType, Dist2DistArtificialValues
from typing import Any, Dict, List, Optional, Tuple, Type
from ._model import Network, UpNetwork
from ._utils import whitebalance
from .typing import CachedData, DataBounders, BoundedAnalyte, Loss, GroupedValues, GroupedBoundedAnalyte
from sweep import DataModule


os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".
from wandb.wandb_run import Run
import wandb


class AutoEncoderDataModule(LightningDataModule):
    def __init__(self, *, batch_size: int, dataset_root_dir: str, intervals_number: int, real_data_percent: float, artificial_data_percent: float, values_distribution: str, **kwargs: Any):
        super().__init__()
        self.datamodule = DataModule(batch_size=batch_size, dataset_root_dir=dataset_root_dir, **kwargs)
        self.dataset_root_dir = dataset_root_dir
        self.dist2dist_cache_dir = os.path.join(dataset_root_dir, "dist2dist")
        self.intervals_number = intervals_number
        self.interval_size = None
        self.intervals_min_value = None
        self.intervals_max_value = None
        self.values_dist = values_distribution
        self.real_data_percent = real_data_percent
        self.artificial_data_percent = artificial_data_percent
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()
        self.prepare_data_any(stage=LightningStage.TRAIN)
        self.prepare_data_any(stage=LightningStage.VALIDATION)
        self.calibrated_pmf_shape = self._get_calibrated_pmf_shape()

    def prepare_data_any(self, stage: str) -> None:
        subset = self.datamodule._load_ready_to_use_subset(stage)
        dims = len(subset[0][0].shape)
        if not self._exists_bounded_analyte_cache(stage=stage):
            match (self.values_dist):
                case Dist2DistArtificialValues.RANDOM:
                    self.prepare_random_data_any(stage=stage, subset=subset)
                case Dist2DistArtificialValues.EQUIDISTANT:
                    self.prepare_equidistant_data_any(stage=stage, subset=subset)
                case Dist2DistArtificialValues.PERMUTED_EQUIDISTANT:
                    self.prepare_permuted_equidistant_data_any(stage=stage, subset=subset)
                case Dist2DistArtificialValues.CYCLIC:
                    self.prepare_cyclic_data_any(stage=stage, subset=subset)
                case _:
                    raise(Exception("Value distribution mode not implemented!"))
        if not self._exists_afine_parameters(stage=stage) and dims>1:
            # Computing afine parameters transformation of the real calibrated pmfs
            afine_parameters = list(map(lambda s: compute_affine_parameters(input=s[0]),subset))
            # Store afine parameters in cache
            self._store_in_cache_afine_parameters(data=afine_parameters, stage=stage)

    def prepare_cyclic_data_any(self, stage: str,  subset: TensorDataset) -> None:
        # Sorts the data
        _, expected_values = subset.tensors
        ord_expected_values, idcs_expected_values = expected_values.sort()
        # Groups the data into GroupedValues
        grouped_values = self._parse_data_to_grouped_values(values=ord_expected_values, indexes=idcs_expected_values)
        # Define the artificial PMFs candidates
        ranges = self._get_candidate_ranges(data=[gv['value'] for gv in grouped_values])
        candidats_in_ranges = list(map(lambda x: self._calculate_candidates_in_range(rng=x), ranges))
        # Selects the indeces of lowers and uppers
        ls_ups_indexes = list(map(lambda lowers, uppers: self._get_lowers_and_uppers_in_group(lowers, uppers), 
                [gv['indexes'] for gv in grouped_values[:-1]],
                [gv['indexes'] for gv in grouped_values[1:]],
        ))
        # Convert the artificial PMF candidates into BoundedAnalyte to store
        artificial_bounded_analytes = list(map(lambda rng, cndt, ls_ups: self._parse_grouped_candidates_to_bounded_analytes(range=rng, candidats=cndt, lowers_uppers=ls_ups), ranges, candidats_in_ranges, ls_ups_indexes))
        artificial_bounded_analytes = [x for xs in artificial_bounded_analytes for x in xs]
        # Transforms real grouped values into list of lists of Bounded Analytes
        real_bounded_analytes = self._parse_grouped_values_to_bounded_analytes(data=grouped_values)
        real_grouped_bounded_analytes = self._create_grouped_bounded_analytes(data=real_bounded_analytes)
        real_bounded_analytes = [ls for lss in real_bounded_analytes for ls in lss]
        # Stores in cache the list of Bounded Analyte ready to be used in the Generator trainig
        self._store_in_cache_bounded_analyte(data=real_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.TRAINING)
        self._store_in_cache_grouped_bounded_analyte(data=real_grouped_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.TRAINING)
        # Stores in cache the list of Bounded Analyte ready to be used for augmentation
        self._store_in_cache_bounded_analyte(data=artificial_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.AUGMENTATION)
   
    def prepare_permuted_equidistant_data_any(self, stage: str, subset: TensorDataset) -> None:
        # Sorts the data
        _, expected_values = subset.tensors
        ord_expected_values, idcs_expected_values = expected_values.sort()
        # Groups the data into GroupedValues
        grouped_values = self._parse_data_to_grouped_values(values=ord_expected_values, indexes=idcs_expected_values)
        # Define the artificial PMFs candidates
        ranges = self._get_candidate_ranges(data=[gv['value'] for gv in grouped_values])
        candidats_in_ranges = list(map(lambda x: self._calculate_candidates_in_range(rng=x), ranges))
        # Selects the indeces of lowers and uppers
        ls_ups_indexes = list(map(lambda lowers, uppers: self._get_lowers_and_uppers_in_group(lowers, uppers), 
                [gv['indexes'] for gv in grouped_values[:-1]],
                [gv['indexes'] for gv in grouped_values[1:]],
        ))
        # Convert the artificial PMF candidates into BoundedAnalyte to store
        artificial_bounded_analytes = list(map(lambda rng, cndt, ls_ups: self._parse_grouped_candidates_to_bounded_analytes(range=rng, candidats=cndt, lowers_uppers=ls_ups), ranges, candidats_in_ranges, ls_ups_indexes))
        artificial_bounded_analytes = [x for xs in artificial_bounded_analytes for x in xs]
        # Transforms real grouped values into list of lists of Bounded Analytes
        real_bounded_analytes = self._parse_grouped_values_to_bounded_analytes(data=grouped_values)
        real_bounded_analytes = [ls for lss in real_bounded_analytes for ls in lss]
        # Stores in cache the list of Bounded Analyte ready to be used in the Generator trainig
        self._store_in_cache_bounded_analyte(data=real_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.TRAINING)
        # Stores in cache the list of Bounded Analyte ready to be used for augmentation
        self._store_in_cache_bounded_analyte(data=artificial_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.AUGMENTATION)
    
    def prepare_equidistant_data_any(self, stage: str,  subset: TensorDataset) -> None:
        # Sorts the data
        _, expected_values = subset.tensors
        ord_expected_values, idcs_expected_values = expected_values.sort()
        # Groups the data into GroupedValues
        grouped_values = self._parse_data_to_grouped_values(values=ord_expected_values, indexes=idcs_expected_values)
        # Define the artificial PMFs candidates
        ranges = self._get_candidate_ranges(data=[gv['value'] for gv in grouped_values])
        candidats_in_ranges = list(map(lambda x: self._calculate_candidates_in_range(rng=x), ranges))
        # Selects the indeces of lowers and uppers
        ls_ups_indexes = list(map(lambda lowers, uppers: self._get_lowers_and_uppers_in_group(lowers, uppers), 
                [gv['indexes'] for gv in grouped_values[:-1]],
                [gv['indexes'] for gv in grouped_values[1:]],
        ))
        # Convert the artificial PMF candidates into BoundedAnalyte to store
        artificial_bounded_analytes = list(map(lambda rng, cndt, ls_ups: self._parse_grouped_candidates_to_bounded_analytes(range=rng, candidats=cndt, lowers_uppers=ls_ups), ranges, candidats_in_ranges, ls_ups_indexes))
        artificial_bounded_analytes = [x for xs in artificial_bounded_analytes for x in xs]
        # Transforms real grouped values into list of lists of Bounded Analytes
        real_bounded_analytes = self._parse_grouped_values_to_bounded_analytes(data=grouped_values)
        real_grouped_bounded_analytes = self._create_grouped_bounded_analytes(data=real_bounded_analytes)
        real_bounded_analytes = [ls for lss in real_bounded_analytes for ls in lss]
        # Stores in cache the list of Bounded Analyte ready to be used in the Generator trainig
        self._store_in_cache_bounded_analyte(data=real_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.TRAINING)
        self._store_in_cache_grouped_bounded_analyte(data=real_grouped_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.TRAINING)
        # Stores in cache the list of Bounded Analyte ready to be used for augmentation
        self._store_in_cache_bounded_analyte(data=artificial_bounded_analytes, stage=stage, folder_name=Dist2DistCacheType.AUGMENTATION)
   
    def prepare_random_data_any(self, stage: str, subset: TensorDataset) -> None:
        # Transforms ready to use data into cached data
        cached_data = self._parse_ready_to_use_data_to_cached_data(subset=subset)
        # Find lower and upper bounders
        bounders = [self._find_valid_lower_and_upper(target=d['value'], cached_data=cached_data) for d in cached_data]
        # Join targets with respective bounders and removing from training data targets with no bounders
        bounded_analyte = [BoundedAnalyte(target=t, bounders=b, interpolation_factor=calculate_interpolation_factor(a=b['lower']['value'], b=b['upper']['value'], x=t['value'])) for t, b in zip(cached_data, bounders) if b['lower']!=None or b['upper']!=None]
        # Calculates the necessary size threshold
        self._calculate_interval_size(data=bounded_analyte)
        # Generate candidates to create artificial data and sort it with the real data
        augmented_data_cache = self._generate_complementary_artificial_data(data=bounded_analyte)
        # Stores in cache the list of Bounded Analyte ready to be used for augmentation
        self._store_in_cache_bounded_analyte(data=augmented_data_cache, stage=stage, folder_name=Dist2DistCacheType.AUGMENTATION)
        # Stores in cache the list of Bounded Analyte ready to be used in the Generator trainig
        self._store_in_cache_bounded_analyte(data=bounded_analyte, stage=stage, folder_name=Dist2DistCacheType.TRAINING)  

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_subset = self._load_ready_to_use_subset(LightningStage.TRAIN, training_mode=True)
            self.val_subset = self._load_ready_to_use_subset(LightningStage.VALIDATION, training_mode=True)
        elif stage == "validate":
            self.val_subset = self._load_ready_to_use_subset(LightningStage.VALIDATION, training_mode=True)
        elif stage == "test":
            self.test_subset = self._load_ready_to_use_subset(LightningStage.TEST, training_mode=True)

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=min(2, multiprocessing.cpu_count()), shuffle=True, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self) -> Any:
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=min(2, multiprocessing.cpu_count()), shuffle=False, pin_memory=True, drop_last=False, persistent_workers=True)

    def test_dataloader(self) -> Any:
        return DataLoader(self.test_subset, batch_size=self.batch_size, num_workers=min(2, multiprocessing.cpu_count()), shuffle=False, pin_memory=True, drop_last=False, persistent_workers=True)

    @classmethod
    def wandb_parameters(cls) -> Dict[str, Any]:
        return {"batch_size": {"distribution": "int_uniform", "min": 5, "max": 30}}

    def artificial_pmf_tensor_dataset(self, stage: str) -> Dataset:
        return self._load_ready_to_use_subset(stage=stage, training_mode=False)

    def _load_ready_to_use_subset(self, stage:str, training_mode: bool=False) -> Dataset:
        subset = self.datamodule._load_ready_to_use_subset(stage)
        if training_mode:
            bounded_analyte = self._load_bounded_analyte_from_cache(stage=stage, folder_name=Dist2DistCacheType.TRAINING)
            dims = len(subset[0][0].shape)
            
            if dims > 1:
                afine_parameters = self._load_afine_parameters_from_cache(stage=stage)
                
            match self.values_dist:
                case Dist2DistArtificialValues.RANDOM:
                    return ItemwiseDataset(bounded_analyte=bounded_analyte, subset=subset, afine_parameters=afine_parameters if dims>1 else None, dims=dims)
                case Dist2DistArtificialValues.EQUIDISTANT:
                    grouped_bounded_analyte = self._load_grouped_bounded_analyte_from_cache(stage=stage, folder_name=Dist2DistCacheType.TRAINING)
                    return GroupwiseDataset(bounded_analyte=bounded_analyte, subset=subset, afine_parameters=afine_parameters if dims>1 else None, grouped_bounded_analyte=grouped_bounded_analyte, dims=dims)
                case Dist2DistArtificialValues.PERMUTED_EQUIDISTANT:
                    return ItemwiseDataset(bounded_analyte=bounded_analyte, subset=subset, afine_parameters=afine_parameters if dims>1 else None, dims=dims)
                case Dist2DistArtificialValues.CYCLIC:
                    grouped_bounded_analyte = self._load_grouped_bounded_analyte_from_cache(stage=stage, folder_name=Dist2DistCacheType.TRAINING)
                    return CyclicDataset(bounded_analyte=bounded_analyte, subset=subset, afine_parameters=afine_parameters if dims>1 else None, grouped_bounded_analyte=grouped_bounded_analyte, dims=dims)
                case _:
                    raise Exception(f"Dataset not maped for {self.values_dist}")
        else:
            bounded_analyte = self._load_bounded_analyte_from_cache(stage=stage, folder_name=Dist2DistCacheType.AUGMENTATION)
            return GeneratedDataset(bounded_analyte=bounded_analyte, subset=subset)

    def _parse_ready_to_use_data_to_cached_data(self, subset: TensorDataset) -> List[CachedData]:
        """Transforms the ready to use data into a list of cached data. `CachedData` keeps the analyte value, index of the in cached saved data and if it is real or artificial data.

        Parameters
        ----------
        subset : TensorDataset
            subset saved preprocessed and saved in cache.

        Returns
        ----------
        List[CachedData]
            a list containing analyte values and indices of the data saved in disck.
        """
        _, expected_values = subset.tensors
        ord_expected_values, idcs_expected_values = expected_values.sort()
        return list(map(lambda v, i: CachedData(value=v, index=i, real=True),ord_expected_values, idcs_expected_values))

    def _find_valid_lower_and_upper(self, target: float, cached_data: List[CachedData]) -> DataBounders:
        """Finds compared to `target` an immediately lower and upper analyte values.

        Parameters
        ----------
        target : float
            a value to use as objective of the search.
        cached_data : List[CachedData]
            the list of cached data maped as `CachedData`.

        Returns
        ----------
        DataBounders
            closest lower and upper values of the taget.
        """
        lowers = [d for d in cached_data if d['value'] < target]
        uppers = [d for d in cached_data if d['value'] > target]
        if len(lowers) == 0 or len(uppers) == 0:
            return DataBounders(lower=None, upper=None)
        return DataBounders(lower=lowers[len(lowers)-1], upper=uppers[0])

    def _generate_complementary_artificial_data(self, data: List[BoundedAnalyte]) -> List[BoundedAnalyte]:
        """Separates the initial data into intervals accordingly to `self.interval_size`, removes randomly samples if the `self.real_data_percent` is smaller than 1, and adds artificial data candidates if `self.artificial_data_percent` is bigger the 0.

        Parameters
        ----------
        data : List[BoundedAnalyte]
            a list of data to be partitioned and manipulated.

        Returns
        ----------
        List[BoundedAnalyte]
            a list of real cached data and artificial data candidates in ascending order.
        """
        interval_values = self._split_data_into_intervals(data)
        mins_maxs = [(iv[0]['target']['value'], iv[len(iv)-1]['target']['value']) for iv in interval_values]

        if self.real_data_percent < 1:
            percent_to_remove = 1 - self.real_data_percent
            remove_n_samples = [round(len(iv)*perc) for iv, perc in zip(interval_values, len(interval_values) * [percent_to_remove])]
            chosen_idxs = [self._chose_k_unique_indexes(value=[v['target']['value'] for v in iv], k=n) for iv, n in zip(interval_values, remove_n_samples)]
            for idxs, ivs in zip(chosen_idxs, interval_values):
                idxs.sort(reverse=True)
                for idx in idxs:
                    del ivs[idx]
        else:
            chosen_idxs = [[] for iv in interval_values]

        if self.artificial_data_percent > 0:
            add_n_samples = [len(ls) for ls in chosen_idxs]
            if sum(add_n_samples) == 0:
                add_n_samples = [int(len(ls)*self.artificial_data_percent) for ls in interval_values]

            for ivs, n, mm in zip(interval_values, add_n_samples, mins_maxs):
                if len(ivs) != 0:
                    values = self._find_new_unique_value(data=ivs, n=n, min_max=mm)
                    bounders = [self._find_lower_and_upper_from_bounded_analyte(v, d) for v, d in zip(values, len(values)*[data])]
                    ivs += [
                        BoundedAnalyte(target=CachedData(value=v, index=-1, real=False), bounders=b, interpolation_factor=calculate_interpolation_factor(a=b['lower']['value'], b=b['upper']['value'], x=v))
                    for v, b in zip(values, bounders)]
                    ivs.sort(key= lambda x: x['target']['value'])

        intervaled_sorted_new_data = [iv for ivs in interval_values for iv in ivs]

        return intervaled_sorted_new_data

    def _find_lower_and_upper_from_bounded_analyte(self, target: float, cached_bounded_analytes: List[BoundedAnalyte]) -> DataBounders:
        """Finds compared to `target` an immediately lower and upper analyte values from `bounders.lower` and `bounders.upper` respectively.

        Parameters
        ----------
        target : float
            a value to use as objective of the search.
        cached_bounded_analytes : List[BoundedAnalyte]
            the list of cached data maped as `BoundedAnalyte`.

        Returns
        ----------
        DataBounders
            closest `bounders.lower` and `bounders.upper` values form the taget.
        """
        lowers = [d for d in cached_bounded_analytes if d['target']['value'] < target]
        uppers = [d for d in cached_bounded_analytes if d['target']['value'] > target]
        if len(lowers) == 0 or len(uppers) == 0:
            assert Exception('Was not possible to find a lower or upper bounder for a target value!')
        return DataBounders(lower=lowers[len(lowers)-1]['bounders']['lower'], upper=uppers[0]['bounders']['upper'])

    def _split_data_into_intervals(self, data: List[BoundedAnalyte]) -> List[List[BoundedAnalyte]]:
        """Separates a list of data into a list of lists accordingly to a number desired intervals (`self.intervals_number`).

        Parameters
        ----------
        data : List[BoundedAnalyte]
            a list of data to organize in sections.

        Returns
        ----------
        List[List[BoundedAnalyte]]
            a list of lists of data.
        """
        interval_idxs = list(range(1, self.intervals_number+1))
        return [self._select_intervaled_data(data=data, ceil=self.intervals_min_value+(i*k)) for i, k in zip(interval_idxs, [self.interval_size] * len(interval_idxs))]

    def _select_intervaled_data(self, data: List[BoundedAnalyte], ceil: float, floor: float = None) -> List[BoundedAnalyte]:
        """Selects values from a list accordingly to a informed interval.

        Parameters
        ----------
        data : List[BoundedAnalyte]
            a list of data to search the desired interval.

        Returns
        ----------
        List[BoundedAnalyte]
            a list of data inside the interval.
        """
        if floor == None:
            floor = ceil - self.interval_size

        return [d for d in data if d['target']['value'] >= floor and d['target']['value'] < ceil]

    def _chose_k_unique_indexes(self, value: List[float], k: int=1) -> List[int]:
        """Choses from a list of indexes k unique values.

        Parameters
        ----------
        last_index : int
            the last index for a list.
        k : int
            the number of indexes to be chosen.

        Returns
        ----------
        List[int]
            a list of unique indexes.
        """
        idxs = list(range(len(value)))
        chosen = []

        while len(chosen) < k:
            i = random.choice(idxs)
            if i not in chosen:
                chosen.append(i)
                idxs.remove(i)

        return chosen

    def _find_new_unique_value(self, data: List[BoundedAnalyte], min_max: Tuple, n: int = 1) -> List[int]:
        """Finds n values that does not exists into the informed data.

        Parameters
        ----------
        data : List[BoundedAnalyte]
            a data list that already exists.
        n : int
            the number of values to be generated.

        Returns
        ----------
        List[int]
            a list of new unique possible values.
        """
        found: float = []
        min, max = min_max
        data_values = [v["target"]["value"] for v in data]
        while len(found) < n:
            value = random.uniform(min, max)
            if (data_values+found).count(value) == 0:
                found.append(value)
        return found

    def _calculate_interval_size(self, data: List[BoundedAnalyte]) -> None:
        """Calculates from a list of `BoundedAnalyte` min, max and the interval value contained in a range.

        Parameters
        ----------
        data : List[BoundedAnalyte]
            quantity of necessary intervals.
        """
        data_size = len(data)
        self.intervals_min_value = data[0]['target']['value'].item()
        self.intervals_max_value = data[data_size-1]['target']['value'].item()
        self.interval_size = (self.intervals_max_value - self.intervals_min_value) / self.intervals_number

    def _store_in_cache_bounded_analyte(self, data: List[BoundedAnalyte], stage: str, folder_name: str) -> None:
        # Gets information about data type
        data_size = len(data)
        nbytes_float32 = torch.finfo(torch.float32).bits // 8
        nbytes_int32 = torch.iinfo(torch.int32).bits // 8
        # Creates cache folder if does not exists
        self._check_if_cache_folder_exists(sub_folder=folder_name)
        folder_path = os.path.join(self.dist2dist_cache_dir, folder_name)
        # Creates target UntypedStorage
        target_value = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-target_value.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        target_index = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-target_index.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        target_real = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-target_real.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        # Creates lower UntypedStorage
        lower_value = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-lower_value.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        lower_index = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-lower_index.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        lower_real = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-lower_real.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        # Creates upper UntypedStorage
        upper_value = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-upper_value.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        upper_index = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-upper_index.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        upper_real = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-upper_real.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        # Creates interpolation factor UntypedStorage
        interpolation_factor = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-interpolation_factor.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        # Storing data into disk
        for index, item in enumerate(tqdm(iter(data), total=data_size, desc=f'Writing "{stage}" split to disk', leave=False)):
            target_value[index] = torch.as_tensor(item['target']['value'], dtype=torch.float32)
            target_index[index] = torch.as_tensor(item['target']['index'], dtype=torch.int32)
            target_real[index] = torch.as_tensor(item['target']['real'], dtype=torch.int32)
            lower_value[index] = torch.as_tensor(item['bounders']['lower']['value'], dtype=torch.float32)
            lower_index[index] = torch.as_tensor(item['bounders']['lower']['index'], dtype=torch.int32)
            lower_real[index] = torch.as_tensor(item['bounders']['lower']['real'], dtype=torch.int32)
            upper_value[index] = torch.as_tensor(item['bounders']['upper']['value'], dtype=torch.float32)
            upper_index[index] = torch.as_tensor(item['bounders']['upper']['index'], dtype=torch.int32)
            upper_real[index] = torch.as_tensor(item['bounders']['upper']['real'], dtype=torch.int32)
            interpolation_factor[index] = torch.as_tensor(item['interpolation_factor'], dtype=torch.float32)
        # Creating indexing history
        dump_data = [[d['target']['value'].item(), d['target']['index'] if d['target']['index'] == -1 else d['target']['index'].item(), d['target']['real'], d['bounders']['lower']['value'].item(), d['bounders']['lower']['index'].item(), d['bounders']['lower']['real'], d['bounders']['upper']['value'].item(), d['bounders']['upper']['index'].item(), d['bounders']['upper']['real'], d['interpolation_factor'].item()] for d in data]
        dump_data.insert(0, ['target_value', 'target_index', 'target_real', 'lower_value', 'lower_index', 'lower_real', 'upper_value', 'upper_index', 'upper_real', 'interpolation_factor'])
        np.savetxt(os.path.join(folder_path, f'{stage}-history.csv'), dump_data, delimiter=", ", fmt="% s")
        # Creating UntypedStorage rebiuld info
        with open(os.path.join(folder_path, f'{stage}-bounded_analyte.json'), "w") as fout:
            json.dump({"num_samples": data_size}, fout)

    def _load_bounded_analyte_from_cache(self, stage:str, folder_name: str) -> List[BoundedAnalyte]:
        folder_path = os.path.join(self.dist2dist_cache_dir, folder_name)
        with open(os.path.join(folder_path, f'{stage}-bounded_analyte.json'), "r") as fin:
            header = json.load(fin)
        data_size = header['num_samples']
        nbytes_float32 = torch.finfo(torch.float32).bits // 8
        nbytes_int32 = torch.iinfo(torch.int32).bits // 8
        # Loading target UntypedStorage
        target_value = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-target_value.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        target_index = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-target_index.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        target_real = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-target_real.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        # Loading lower UntypedStorage
        lower_value = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-lower_value.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        lower_index = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-lower_index.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        lower_real = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-lower_real.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        # Loading upper UntypedStorage
        upper_value = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-upper_value.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        upper_index = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-upper_index.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        upper_real = IntTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-upper_real.bin"), nbytes=(nbytes_int32*data_size), shared=True))
        # Loading interpolation factor UntypedStorage
        interpolation_factor = FloatTensor(UntypedStorage.from_file(filename=os.path.join(folder_path, f"{stage}-interpolation_factor.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        # Casting raw torch data into Bounded Analyte
        bounded_analytes = list()
        for index in tqdm(range(data_size), total=data_size, desc=f'Casting "{stage}" raw torch data into Bounded Analyte', leave=False):
            bounded_analytes.append(BoundedAnalyte(
                target=CachedData(value=target_value[index].item(), index=target_index[index].item(), real=bool(target_real[index].item())),
                bounders=DataBounders(
                    lower=CachedData(value=lower_value[index].item(), index=lower_index[index].item(), real=bool(lower_real[index].item())),
                    upper=CachedData(value=upper_value[index].item(), index=upper_index[index].item(), real=bool(upper_real[index].item()))
                ),
                interpolation_factor=interpolation_factor[index].item()
            ))
        return bounded_analytes

    def _exists_bounded_analyte_cache(self, stage:str) -> bool:
        filenames = [f"{stage}-target_value.bin", f"{stage}-target_index.bin", f"{stage}-target_real.bin", f"{stage}-lower_value.bin", f"{stage}-lower_index.bin", f"{stage}-lower_real.bin", f"{stage}-upper_value.bin", f"{stage}-upper_index.bin", f"{stage}-upper_real.bin"]
        folders_path = [os.path.join(self.dist2dist_cache_dir, foldername) for foldername in [Dist2DistCacheType.AUGMENTATION, Dist2DistCacheType.TRAINING]]
        exists = list()
        for fp in folders_path:
            exists += list(map(lambda filename: os.path.exists(os.path.join(fp, filename)), filenames))

        return not exists.__contains__(False)

    def _store_in_cache_grouped_bounded_analyte(self, data: List[GroupedBoundedAnalyte], stage: str, folder_name: str) -> None:
        # Creates cache folder if does not exists
        self._check_if_cache_folder_exists(sub_folder=folder_name)
        folder_path = os.path.join(self.dist2dist_cache_dir, folder_name)
        # Stores data into JSON
        with open(os.path.join(folder_path, f'{stage}-index_permutation.json'), "w") as fout:
            json.dump({"data": data}, fout)
        
    def _load_grouped_bounded_analyte_from_cache(self, stage:str, folder_name:str) -> List[GroupedBoundedAnalyte]:
        folder_path = os.path.join(self.dist2dist_cache_dir, folder_name)
        with open(os.path.join(folder_path, f'{stage}-index_permutation.json'), "r") as fin:
            data = json.load(fin)['data']
        return data

    def _store_in_cache_afine_parameters(self, data:Tuple[np.ndarray, np.ndarray, float, np.ndarray], stage:str) -> None:
        # Gets information about data type
        data_size = len(data)
        nbytes_float32 = torch.finfo(torch.float32).bits // 8
        centers_shape = data[0][0].shape
        translations_shape = data[0][1].shape
        transformed_shape = data[0][3].shape
        # Creates cache folder if does not exists
        self._check_if_cache_folder_exists()
        # Creates afine parameters UntypedStorage
        angles = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-angles.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        centers = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-centers.bin"), nbytes=(np.prod(centers_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *centers_shape)
        translations = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-translations.bin"), nbytes=(np.prod(translations_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *translations_shape)
        transformed = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-transformed.bin"), nbytes=(np.prod(transformed_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *transformed_shape)
        # Storing data into disk
        for index, item in enumerate(tqdm(iter(data), total=data_size, desc=f'Writing "{stage}" afine transformations to disk', leave=False)):
            centers[index, ...] = torch.as_tensor(item[0], dtype=torch.float32)
            translations[index, ...] = torch.as_tensor(item[1], dtype=torch.float32)
            angles[index] = torch.as_tensor(item[2], dtype=torch.float32)
            transformed[index, ...] = torch.as_tensor(item[3], dtype=torch.float32)
        # Creating UntypedStorage rebiuld info
        with open(os.path.join(self.dist2dist_cache_dir, f'{stage}-afine_parameters.json'), "w") as fout:
            json.dump({"num_samples": data_size, "centers_shape": centers_shape, "translations_shape": translations_shape, "transformed_shape": transformed_shape}, fout)

    def _load_afine_parameters_from_cache(self, stage:str) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        with open(os.path.join(self.dist2dist_cache_dir, f'{stage}-afine_parameters.json'), "r") as fin:
            header = json.load(fin)
        data_size = header['num_samples']
        nbytes_float32 = torch.finfo(torch.float32).bits // 8
        centers_shape = header['centers_shape']
        translations_shape = header['translations_shape']
        transformed_shape = header['transformed_shape']
        # Creates afine parameters UntypedStorage
        angles = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-angles.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        centers = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-centers.bin"), nbytes=(np.prod(centers_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *centers_shape)
        translations = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-translations.bin"), nbytes=(np.prod(translations_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *translations_shape)
        transformed = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-transformed.bin"), nbytes=(np.prod(transformed_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *transformed_shape)
        # Casting raw torch data into Tuple[np.ndarray, np.ndarray, float, np.ndarray]
        afine_parameters = list()
        for index in tqdm(range(data_size), total=data_size, desc=f'Casting "{stage}" raw torch data into Tuple', leave=False):
            afine_parameters.append(tuple([centers[index], translations[index], angles[index], transformed[index]]))
        return afine_parameters

    def _store_in_cache_artificial_pmfs(self, stage:str, data:List[Tuple[torch.Tensor, torch.Tensor]]):
        # Gets information about data type
        data_size = len(data)
        nbytes_float32 = torch.finfo(torch.float32).bits // 8
        pmfs_shape = data[0][0].shape
        # Creates cache folder if does not exists
        self._check_if_cache_folder_exists()
        # Creates afine parameters UntypedStorage
        pmfs = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-artificial_pmfs.bin"), nbytes=(np.prod(pmfs_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *pmfs_shape)
        expected_values = FloatTensor(UntypedStorage.from_file(filename=os.path.join(self.dist2dist_cache_dir, f"{stage}-expected_values.bin"), nbytes=(nbytes_float32*data_size), shared=True))
        # Storing data into disk
        for index, item in enumerate(tqdm(iter(data), total=data_size, desc=f'Writing "{stage}" artificial pmfs and expected values to disk', leave=False)):
            pmf, value = item
            pmfs[index, ...] = pmf
            expected_values[index, ...] = value
        # Creating UntypedStorage rebiuld info
        with open(os.path.join(self.dist2dist_cache_dir, f'{stage}-artificial_pmfs.json'), "w") as fout:
            json.dump({"num_samples": data_size, "pmfs_shape": pmfs_shape}, fout)

    def _exists_afine_parameters(self, stage:str) -> bool:
        filenames = [f"{stage}-angles.bin", f"{stage}-centers.bin", f"{stage}-translations.bin", f"{stage}-transformed.bin"]
        exists = list(map(lambda filename: os.path.exists(os.path.join(self.dist2dist_cache_dir, filename)), filenames))
        return not exists.__contains__(False)

    def _check_if_cache_folder_exists(self, sub_folder: str=None) -> None:
        if os.path.isdir(self.dist2dist_cache_dir) == False:
            os.mkdir(self.dist2dist_cache_dir)
        if sub_folder is not None and os.path.isdir(os.path.join(self.dist2dist_cache_dir, sub_folder)) == False:
            os.mkdir(os.path.join(self.dist2dist_cache_dir, sub_folder))

    def _get_calibrated_pmf_shape(self) -> Tuple[int,int]:
        filepath = os.path.join(self.dataset_root_dir, 'train-processed_samples.json')
        if os.path.exists(filepath):
            with open(os.path.join(filepath), "r") as fin:
                header = json.load(fin)
                return tuple(map(lambda x: int(x), header['calibrated_pmf_shape']))
        else:
            raise Exception("train-processed_samples.json not found")

    def _parse_data_to_grouped_values(self, values: Tensor, indexes: Tensor) -> List[GroupedValues]:
        """Transforms the passed oredered values and indeces in a `List[GroupedValues]`.

        Parameters
        ----------
        data : Tensor
            a tensor of ordened values.
        indexes : Tensor
            a tensor of indexes refering the stored cache data.

        Returns
        ----------
        List[GroupedValues]
            a list of grouped values contaning value, group label and a list of indexes correspondinto to the associated value.
        """
        cur_value, count, res = None, -1, list()
        for value, index in zip(values, indexes):
            if cur_value != value:
                cur_value = value
                count += 1
                res.append(GroupedValues(
                    value=value,
                    group=count,
                    indexes=[index]
                ))
            else:
                last_indx = len(res)-1
                res[last_indx]['indexes'].append(index)
        return res

    def _get_candidate_ranges(self, data: List[Tensor]) -> List[Tuple[Tensor,Tensor]]:
        """From a ordered list returns a list of tuple indicatin every possible range between values.

        Parameters
        ----------
        data : List[Tensor]
            a ordered list of tensors.

        Returns
        ----------
        List[Tuple[Tensor,Tensor]]
            a list of ranges as tuples.
        """
        return [(mn, mx) for mn, mx in zip(data[:-1], data[1:])]
    
    def _calculate_candidates_in_range(self, rng: Tuple[Tensor, Tensor]) -> List[Tensor]:
        """Calculate N equidistant values in a range.

        Parameters
        ----------
        rng : Tuple[Tensor, Tensor]
            a tuple containing a range, thus, the firt element is necessarily smaller then the second element

        Returns
        ----------
        List[float]
            a list of values equidistants in a especific range.
        """
        N = int(self.artificial_data_percent)+1
        step_size = (rng[1]-rng[0])/N
        return [rng[0]+i*step_size for i in range(N)][1:]
    
    def _get_lowers_and_uppers_in_group(self, lowers: List[Tensor], uppers: List[Tensor]) -> List[Tuple[Tensor, Tensor]]:
        """From a list of lowers and uppers candidates indexes, choses the PMF from wich the artificial ones will be generated.

        Parameters
        ----------
        lowers : List[Tensor]
            a List of indexes coresponding to every lower candidate.
        uppers : List[Tensor]
            a List of indexes coresponding to every upper candidate.

        Returns
        ----------
        List[Tuple[Tensor, Tensor]]
            a list of tuple containing a pair of real PMF indexes to be use as lower and upper of a artificial PMF.
        """
        ls, ups, res = lowers, uppers, list()
        for _ in range(int(self.artificial_data_percent)):
            if len(ls) == 0: ls = lowers
            if len(ups) == 0: ups = uppers
            random.shuffle(ls)
            random.shuffle(ups)
            res.append((ls[0], ups[0]))
            ls, ups = ls[1:], ups[1:]
        return res
    
    def _parse_grouped_candidates_to_bounded_analytes(self, range: Tuple[Tensor,Tensor], candidats: List[Tensor], lowers_uppers: List[Tuple[Tensor, Tensor]]) -> List[BoundedAnalyte]:
        """Transforms the artificial candidats, ranges, lowers and uppers candidats in bounded analytes.

        Parameters
        ----------
        range : Tuple[Tensor, Tensor]
            a Tuple containing the lower and upper PMF's values used as source for the interpolation
        candidats : List[Tensor]
            a List of values candidats to be used to generate artificial PMFs
        lowers_uppers : List[Tuple[Tensor, Tensor]]
            a List of Tuples containing lowers and uppers indexes permutations, making it possible to arrange candidates without bies

        Returns
        ----------
        List[BoundedAnalyte]
            a list of biulded bounded analytes
        """
        targets = [CachedData(value=cnd, index=-1, real=False) for cnd in candidats]
        lowers = [CachedData(value=value, index=indx[0], real=False) for indx, value in zip(lowers_uppers, [range[0]]*len(lowers_uppers))]
        uppers = [CachedData(value=value, index=indx[1], real=False) for indx, value in zip(lowers_uppers, [range[1]]*len(lowers_uppers))]
        inter_factors = [calculate_interpolation_factor(a=float(rng[0]), b=float(rng[1]), x=cnd) for cnd, rng in zip(candidats, [range]*len(candidats))]
        return [BoundedAnalyte(target=tg, bounders=DataBounders(lower=lw, upper=up), interpolation_factor=itp) for tg, lw, up, itp in zip(targets, lowers, uppers, inter_factors)]
    
    def _parse_grouped_values_to_bounded_analytes(self, data: List[GroupedValues]) -> List[List[BoundedAnalyte]]:
        """Transforms a list of `GoupedValue` into a list of lists of `BoundedAnalyte`

        Parameters
        ----------
        data : List[GroupedValues]
            a list of ordered grouped values
        
        Returns
        ----------
        List[List[BoundedAnalyte]]
            a list of lists of ordered bounded analytes
        """
        return [self._get_every_index_permutation(target=t, lower=l, upper=u) for t, l, u in zip(data[1:-1], data[:-2], data[2:])]
    
    def _get_every_index_permutation(self, target: GroupedValues, lower: GroupedValues, upper: GroupedValues) -> List[BoundedAnalyte]:
        """Gets every candidate index permutation of target, lower and upper.

        Parameters
        ----------
        target : GroupedValues
            a list of ordered grouped values target candidats
        lower : GroupedValues
            a list of ordered grouped values lower candidats
        upper : GroupedValues
            a list of ordered grouped values upper candidats
        
        Returns
        ----------
        List[BoundedAnalyte]
            a list of ordered bounded analytes
        """
        lower_upper_indxs = list(itertools.product(lower['indexes'], upper['indexes']))
        target_lower_upper_indxs = list(itertools.product(target['indexes'], lower_upper_indxs))
        inter_fact = calculate_interpolation_factor(a=lower['value'], b=upper['value'], x=target['value'])
        return list(map(lambda x: BoundedAnalyte(
            target=CachedData(value=target['value'], index=x[0], real=True),
            bounders=DataBounders(
                lower=CachedData(value=lower['value'], index=x[1][0], real=True),
                upper=CachedData(value=upper['value'], index=x[1][1], real=True),
            ),
            interpolation_factor=inter_fact
        ),target_lower_upper_indxs))
    
    def _create_grouped_bounded_analytes(self, data: List[List[BoundedAnalyte]]) -> List[GroupedBoundedAnalyte]:
        """Transforms a list of lists of Bounded Analyte into a list of Grouped Bounded Analytes.

        Parameters
        ----------
        data : List[List[BoundedAnalyte]]
            a list of lists of ordered Bounded Analytes
        
        Returns
        ----------
        List[GroupedBoundedAnalyte]
            a list of ordered grouped bounded analytes
        """
        indx = 0
        res = list()
        for d in data:
            size = len(d)
            res.append(GroupedBoundedAnalyte(value=d[0]['target']['value'].item(), indexes=[i+indx for i in range(size)]))
            indx += size
        return res
    

class BaseAutoEncoder(ABC, LightningModule):
    def __init__(self, *, early_stopping_patience: int, learning_rate: float, learning_rate_patience: int, weight_decay: float, regressor: Type[Network], generator: Type[UpNetwork], calibrated_pmf_shape: Tuple[int,...], dev_exec_mode: str=DevMode.PROD, dev_log_frequence:int=10, **kwargs: Any) -> None:
        super().__init__()
        # Keep the input arguments.
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.weight_decay = weight_decay
        self.regressor = regressor
        self.regressor.eval()
        self.generator = generator
        self.calibrated_pmf_shape = calibrated_pmf_shape
        self.dev_exec_mode = dev_exec_mode
        self.dev_log_freq = dev_log_frequence
        self.inf_vs_target = {ModelMode.TRAIN: list(), ModelMode.VALIDATION: list(), ModelMode.TEST: list()}

    @abstractmethod
    def _any_epoch_end(self, mode_name: str) -> None:
        raise NotImplementedError  # To be implemented by the subclass.

    @abstractmethod
    def _any_step(self, batch, batch_idx: int, mode_name: str) -> Loss:
        raise NotImplementedError  # To be implemented by the subclass.

    def configure_callbacks(self) -> List[Callback]:
        # Apply early stopping.
        return [
            EarlyStopping(monitor="Loss/Val", mode="min", patience=self.early_stopping_patience),
            LearningRateMonitor(logging_interval='epoch')
        ]

    def configure_optimizers(self) -> Dict[str, Any]:
        # Set the optimizer.
        optimizer = Adam(self.generator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # Set the learning rate scheduler.
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=self.learning_rate_patience)
        # Return the configuration.
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "Loss/Val", "interval": "epoch",}}

    def forward(self, input: Any) -> Any:
        return self.generator(input)

    def on_test_epoch_end(self) -> None:
        self._any_epoch_end(ModelMode.TEST)

    def on_train_epoch_end(self) -> None:
        self._any_epoch_end(ModelMode.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._any_epoch_end(ModelMode.VALIDATION)

    def training_step(self, batch, batch_idx: int) -> Loss:
        return self._any_step(batch, batch_idx, ModelMode.TRAIN)

    def validation_step(self, batch, batch_idx: int) -> None:
        self._any_step(batch, batch_idx, ModelMode.VALIDATION)

    def test_step(self, batch, batch_idx: int) -> None:
        self._any_step(batch, batch_idx, ModelMode.TEST)

    def on_train_end(self) -> None:
        self.on_any_end(ModelMode.TRAIN)
        self.log_last_inferences(ModelMode.TRAIN)
        self.log_last_inferences(ModelMode.VALIDATION)

    def on_validation_end(self) -> None:
        self.on_any_end(ModelMode.VALIDATION)

    def on_test_end(self) -> None:
        self.on_any_end(ModelMode.TEST)

    def on_any_end(self, mode: str) -> None:
        pass

    def log_last_inferences(self, mode):
        predicted_values = self.inf_vs_target[mode][0]['predicted_value']
        target_values = self.inf_vs_target[mode][0]['target_value']
        sorted_target = torch.sort(target_values)[0].tolist()
        wandb.log({
            f"inferences/{mode}" : wandb.plot.line_series(
            xs=sorted_target,
            ys=[torch.sort(predicted_values)[0].tolist(), sorted_target],
            keys=["Inferences", "Targets"],
            title=f"Artificial PMFs Inferences ({mode})",
            xname="Targets")
        })

    @classmethod
    @abstractmethod
    def wandb_metric(cls) -> Dict[str, Any]:
        raise NotImplementedError  # To be implemented by the subclass.

    @classmethod
    def wandb_parameters(cls) -> Dict[str, Any]:
        return {
            "learning_rate": { "distribution": "uniform", "min": 1e-12, "max": 1e-1 },
            "weight_decay": { "distribution": "uniform", "min": 1e-6, "max": 1e-1 },
        }

class ContinuousAutoEncoder(BaseAutoEncoder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Save hyper-parameters to self.hparams. They will also be automatically logged as config parameters in Weights & Biases.
        self.save_hyperparameters(ignore=None)
        # Set the loss function.
        self.criterion = EMDLoss()
        # Set metrics.
        self.metrics = ModuleDict({mode_name: MetricCollection({
            "MAE": MeanAbsoluteError(),
            "MAPE": MeanAbsolutePercentageError(),
        }) for mode_name in [ModelMode.TRAIN, ModelMode.VALIDATION, ModelMode.TEST]})
        # Set hook at the feature extraction ending
        features_layer = self.regressor.model.backbone.features
        self.add_features_hook(features_layer[len(features_layer)-1])
        self.features = list()
        # Set the pad size used for margin to apply linear transformation
        pad = get_padding(roi=self.regressor.input_roi, calibrated_pmf_shape=self.calibrated_pmf_shape)
        if len(pad)==4:
            self.output_regressor_pad = torch.nn.ZeroPad2d(pad)
        else:
            self.output_regressor_pad = torch.nn.ConstantPad1d(pad, 0.0)
        # Set dicts to store outputs
        self.outputs = {ModelMode.TRAIN: list(), ModelMode.VALIDATION: list(), ModelMode.TEST: list()}
        # Set the number of dims to be worked
        self.dims = None

    def _any_epoch_end(self, mode_name: str) -> None:
        metrics: MetricCollection = self.metrics[mode_name]  # type: ignore
        self.log_dict({f'{metric_name}/{mode_name}/Epoch': value for metric_name, value in metrics.compute().items()})
        metrics.reset()

        outputs = self.outputs[mode_name]
        self.log_dict({f"sum_diff_abs/{mode_name}_epoch": torch.stack([x["sum_diff_abs"] for x in outputs]).mean().item()})

        self.inf_vs_target[mode_name].clear()
        self.inf_vs_target[mode_name].append({
            'predicted_value': torch.cat([x["predicted_value"] for x in outputs]).flatten(),
            'target_value': torch.cat([x["target_value"] for x in outputs]).flatten()
        })

        # Cleans the outputs before next epoch
        self.outputs[mode_name].clear()

    def _any_step(self, batch, batch_idx, mode_name: str) -> Loss:
        # TODO Caso seja alterada o recorte da ROI no Encoder, refazer os prints para validar
        target_value, target, lower, upper, inter_factor, trans_center, trans_translate, trans_angle, trans_target = batch
        # Extracts the features from lower and upper PMFs
        with torch.no_grad():
            # TODO passar as lowers e uppers como mini batch
            del self.features[:]
            for _, pmf in enumerate([lower.squeeze(dim=1), upper.squeeze(dim=1)]):
                if self.regressor.training:
                    self.regressor.eval()
                self.regressor(pmf)
            if self.dims is None:
                self.dims = len(self.features[0].shape)
        # Interpolates the lower and upper PMFs
        interpolated_feature = torch.lerp(self.features[0], self.features[1], inter_factor.reshape(tuple([len(inter_factor)]+(self.dims-1)*[1])).to(torch.float32))
        # Generates PMFs from a interpolated feature
        output = self.generator(interpolated_feature)
        output_padded = self.output_regressor_pad(output)
        # Computes loss
        if self.dims==4:
            trans_output = rotate_and_translate_in_batch(A=output_padded, translate=trans_translate, angle=trans_angle, center=trans_center)
            loss = self.criterion(trans_output, trans_target)
        else:
            trans_output = output_padded
            loss = self.criterion(trans_output, target.unsqueeze(dim=1))
        # Logs loss
        self.log(f"Loss/{mode_name}", loss)
        # Logs the chart of marginals for comparition
        if self.dev_exec_mode == DevMode.DEBUG and self.current_epoch%self.dev_log_freq==0:
            roied_target = target[..., self.regressor.input_roi[0][0]:self.regressor.input_roi[0][1]+1, self.regressor.input_roi[1][0]:self.regressor.input_roi[1][1]+1]
            self.log_marginals(mode=mode_name, target=roied_target, trans_target=trans_target, output=output.detach().squeeze(dim=1), trans_output=trans_output.detach().squeeze(dim=1), batch_idx=batch_idx)
        # Predicts from the transformed generator output
        with torch.no_grad():
            if self.regressor.training:
                self.regressor.eval()
            predicted_value, _ = self.regressor(output_padded.detach().squeeze(1))
            sum_diff_abs = abs(predicted_value - target_value).sum()
            metrics: MetricCollection = self.metrics[mode_name]  # type: ignore
            self.log_dict({f'{metric_name}/{mode_name}/Step': value for metric_name, value in metrics(predicted_value, target_value).items()})
            # Logs the artificial PMF output
            if self.dev_exec_mode == DevMode.DEBUG and self.current_epoch%self.dev_log_freq==0:
                self.trainer.logger.experiment.log({
                        f"Outputs_{mode_name}/batch_{batch_idx}": [wandb.Image(out, caption=f"Infered: {round(inf.item(), 4)}") for (inf, out) in zip(predicted_value, output)]
                    })
        # Cleans the extracted features record
        del self.features[:]
        # Logs the Ground Truths and Affine Transformed Groundtruths
        if self.dev_exec_mode == DevMode.DEBUG and self.current_epoch==0:
            self.trainer.logger.experiment.log({
                f"Ground Truth_{mode_name}/batch_{batch_idx}": [wandb.Image(t, caption=f"Anoted Alk.: {round(alk.item(), 4)}") for (alk, t) in zip(target_value, roied_target)]
            })
            self.trainer.logger.experiment.log({
                f"Transformed Ground Truth/{mode_name}/batch_{batch_idx}": [wandb.Image(t, caption=f"Anoted Alk.: {round(alk.item(), 4)}") for (alk, t) in zip(target_value, trans_target)]
            })
        # Keeps outputs for later upload
        self.outputs[mode_name].append({
            "predicted_value": predicted_value,
            "target_value": target_value,
            "sum_diff_abs": sum_diff_abs,
        })

        return loss

    @classmethod
    def wandb_metric(cls) -> Dict[str, Any]:
        return {"name": "sum_diff_abs/Val_epoch", "goal": "minimize"}

    def add_features_hook(self, module: torch.nn.Module):
        module.register_forward_hook(
            lambda layer, _, output: self.features.append(output.detach())
        )

    def log_marginals(self, mode:str, target:torch.Tensor, trans_target:torch.Tensor, output:torch.Tensor, trans_output:torch.Tensor, batch_idx:int):
        o_v, o_h = torch.sum(output, dim=-1, keepdim=True).flatten().tolist(), torch.sum(output, dim=-2, keepdim=True).flatten().tolist()
        t_v, t_h = torch.sum(target, dim=-1, keepdim=True).flatten().tolist(), torch.sum(target, dim=-2, keepdim=True).flatten().tolist()
        trans_o_v, trans_o_h = torch.sum(trans_output, dim=-1, keepdim=True).flatten().tolist(), torch.sum(trans_output, dim=-2, keepdim=True).flatten().tolist()
        trans_t_v, trans_t_h = torch.sum(trans_target, dim=-1, keepdim=True).flatten().tolist(), torch.sum(trans_target, dim=-2, keepdim=True).flatten().tolist()

        wandb.log({
            f"marginals_vertical_{mode}/{batch_idx}" : wandb.plot.line_series(
            xs=[i for i in range(len(t_v))],
            ys=[t_v, o_v],
            keys=["Target Marginal Vertical", "Output Marginal Vertical"],
            title=f"Target vs Output Vertical Marginals ({mode})/batch_{batch_idx}",
            xname="Marginal"),
            f"marginals_horizontal_{mode}/{batch_idx}" : wandb.plot.line_series(
            xs=[i for i in range(len(t_h))],
            ys=[t_h, o_h],
            keys=["Target Marginal Horizontal", "Output Marginal Horizontal"],
            title=f"Target vs Output Horizontal Marginals ({mode})/batch_{batch_idx}",
            xname="Marginal"),
            f"trans_marginals_vertical_{mode}/{batch_idx}" : wandb.plot.line_series(
            xs=[i for i in range(len(trans_t_v))],
            ys=[trans_t_v, trans_o_v],
            keys=["Transformed Target Marginal Vertical", "Transformed Output Marginal Vertical"],
            title=f"Transformed Target vs Output Vertical Marginals ({mode})/batch_{batch_idx}",
            xname="Marginal"),
            f"trans_marginals_horizontal_{mode}/{batch_idx}" : wandb.plot.line_series(
            xs=[i for i in range(len(trans_t_h))],
            ys=[trans_t_h, trans_o_h],
            keys=["Transformed Target Marginal Horizontal", "Transformed Output Marginal Horizontal"],
            title=f"Transformed Target vs Output Horizontal Marginals ({mode})/batch_{batch_idx}",
            xname="Marginal")
        })


def get_padding(roi, calibrated_pmf_shape) -> Tuple[int,...]:
    match len(roi):
        case 1:
            pmf_width = calibrated_pmf_shape[0]
            x1, x2 = roi[0]
            left = x1-1
            width = x2+1-x1
            return (left, pmf_width-(left+width))
        case 2:
            pmf_height, pmf_width = calibrated_pmf_shape
            ys, xs = roi
            left = xs[0]-1
            top = ys[0]-1
            height = ys[1]+1-ys[0]
            width = xs[1]+1-xs[0]
            return (left, pmf_width-(left+width), top, pmf_height-(top+height))
        case _:
            raise Exception(f"get_padding() not implemented for {len(roi)}d!")


def _tracked_run(*, checkpoint_dir: str, gpus: int, auto_encoder_model_class: Type[BaseAutoEncoder], generator_network_class: Type[UpNetwork], oracle_network_class: Type[Network], oracle_checkpoint: str, seed: Optional[int], accelerator:str ='gpu', sweep_mode: str=WandbMode.ONLINE, **kwargs: Any) -> None:
    # Start a new tracked run at Weights & Biases.
    with wandb.init(mode=sweep_mode) as run:
        assert isinstance(run, Run)
        # Ensure full reproducibility.
        if seed is not None:
            pl.seed_everything(seed, workers=True)
        # Setup the data module.
        datamodule = AutoEncoderDataModule(**run.config.as_dict(), **kwargs)
        datamodule.prepare_data()
        calibrated_pmf_shape = datamodule.calibrated_pmf_shape
        # Load the oracle model from checkpoint
        oracle_model = oracle_network_class.load_from_checkpoint(oracle_checkpoint)
        oracle_model.eval()
        # Assembles the generator NN architecture
        roi = oracle_model.input_roi
        c = get_featured_roi_size(regressor=oracle_model, input_size=calibrated_pmf_shape)[1]
        generator_model = generator_network_class(input_roi=roi, in_channels=c)
        # Setup the auto encoder.
        auto_encoder_model = auto_encoder_model_class(
            regressor=oracle_model,
            generator=generator_model,
            input_roi=roi,
            in_channels=c,
            calibrated_pmf_shape=calibrated_pmf_shape,
            **run.config.as_dict(),
            **kwargs
        )
        # Setup the trainer.
        trainer = pl.Trainer(
            logger=WandbLogger(experiment=run),
            accelerator=accelerator,
            devices=gpus,
            default_root_dir=checkpoint_dir,
            log_every_n_steps=10,
            max_epochs=-1,
            num_sanity_val_steps=0,
            gradient_clip_val=0.5,
            callbacks=[
                LearningRateMonitor(logging_interval='epoch'),
            ])
        # Perform fitting.
        trainer.fit(auto_encoder_model, datamodule=datamodule)
        # Save trained model.
        checkpoint_filepath = os.path.join(checkpoint_dir, f'{generator_network_class.__name__}.ckpt')
        trainer.save_checkpoint(checkpoint_filepath, weights_only=True)


def _generate_artificial_pmfs(*, generator_network_class:Type[UpNetwork], oracle_network_class:Type[Network], oracle_checkpoint:str, generator_checkpoint:str, dataset_root_dir:str, batch_size: int=1,**kwargs: Any):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load artificial desired pmfs to be generated
    datamodule = AutoEncoderDataModule(batch_size=batch_size, dataset_root_dir=dataset_root_dir, **kwargs)
    datamodule.prepare_data()
    dataset = datamodule.artificial_pmf_tensor_dataset(stage=LightningStage.TRAIN)
    # Checking if exists artificial pmfs
    if _check_if_artificial_pmfs_exists(stage=LightningStage.TRAIN, dataset_root_dir=dataset_root_dir):
        raise Exception("Artificial PMFs has already been generated!")
    # Load the oracle model from checkpoint
    oracle_model = oracle_network_class.load_from_checkpoint(oracle_checkpoint)
    oracle_model.to(device)
    oracle_model.eval()
    # Adding hook to regressor last feature layer
    features_layer = oracle_model.model.backbone.features
    features = list()
    features_layer[len(features_layer)-1].register_forward_hook(
        lambda layer, _, output: features.append(output.detach())
    )
    # Load the generator model from checkpoint
    generator_model = generator_network_class.load_from_checkpoint(generator_checkpoint)
    generator_model.to(device)
    generator_model.eval()
    # Calculating padding ROI
    pad = get_padding(roi=oracle_model.input_roi, calibrated_pmf_shape=generator_network_class.calibrated_pmf_shape(generator_checkpoint))
    padding = torch.nn.ZeroPad2d(pad) if len(pad)==4 else torch.nn.ConstantPad1d(pad, 0.0)
    # Verifying if artificial pmfs already exists
    dist2dist_cache_dir = os.path.join(dataset_root_dir, "dist2dist")
    filename_calibrated_pmf = f"{LightningStage.TRAIN}-calibrated_pmf"
    filename_expected_value = f"{LightningStage.TRAIN}-expected_value"
    existing_files = [os.path.isfile(os.path.join(dist2dist_cache_dir, fn)) for fn in [filename_calibrated_pmf, filename_expected_value]]
    if existing_files.__contains__(True):
        raise("Already exist artificial pmf generated. If new ones is necessary, delete dist2dist cache folder.")
    # Generate artificial pmfs
    data = list()
    for i in tqdm(range(len(dataset)), desc="Generating Artificial PMFs"):
        expected_value, lower, upper, interp = dataset[i]
        del features[:]
        with torch.no_grad():
            oracle_model(lower.to(device).unsqueeze(0))
            oracle_model(upper.to(device).unsqueeze(0))
            interpolated_feature = torch.lerp(features[0], features[1], torch.tensor(interp, device=device))
            data.append((padding(generator_model(interpolated_feature).squeeze((0,1))), expected_value))
    # Adding original data if the equidistant stategy os was used
    if kwargs['values_distribution'] == Dist2DistArtificialValues.EQUIDISTANT:
        # self.datamodule._load_ready_to_use_subset(stage)
        subset = datamodule.datamodule._load_ready_to_use_subset(LightningStage.TRAIN)
        for i in tqdm(range(len(subset)), desc="Adding Real PMFs"):
            calibrated_pmf, exp_values = subset[i]
            data.append((calibrated_pmf.to(device), float(exp_values)))
    # Saving artificial PMFs in the cache
    datamodule._store_in_cache_artificial_pmfs(stage=LightningStage.TRAIN, data=data)


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