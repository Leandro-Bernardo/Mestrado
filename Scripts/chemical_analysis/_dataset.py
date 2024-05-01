from ._utils import bgr_to_lab, compute_calibrated_pmf, compute_theoretical_value, correct_theoretical_value
from .typing import App, AuxiliarySolution, AuxiliarySolutionComponent, ChamberType, Device, Sample, SolutionComponent, Stock, StockAliquot
from tqdm import tqdm
from abc import abstractmethod
from collections.abc import Sized
from datetime import datetime
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Tuple,  TypeVar, Union
import albumentations
import copy, json, os, shutil, warnings
import cv2
import numpy as np


DATETIME_FORMAT: Final[str] = "%Y.%m.%d %H:%M:%S %z"


CONCENTRATION_UNIT_FROM_FUNCTION: Final[Dict[str, str]] = {
    "ACID": "MOL_PER_LITER",
    "COSOLVENT": "PERCENT",
    "COMPLEXING": "MOL_PER_LITER",
    "DYE": "MILLIGRAM_PER_LITER",
    "ION": "MOL_PER_LITER",
    "LIQUIDATOR": "MOL_PER_LITER",
    "MODIFYING": "PERCENT_WEIGHT_VOLUME",
    "PRECIPITANT": "MOL_PER_LITER",
    "REDUCING": "MOL_PER_LITER",
}


SOLUTION_COMPONENT_FUNCTIONS: Final[Dict[str, str]] = {
    "ACID": "Ácido",
    "COSOLVENT": "Co-Solvente",
    "COMPLEXING": "Complexante",
    "DYE": "Indicador",
    "ION": "Força Iônica",
    "LIQUIDATOR": "Liquidante",
    "MODIFYING": "Modificador",
    "PRECIPITANT": "Precipitante",
    "REDUCING": "Redutor",
}


UNITS: Final[Dict[str, str]] = {
    "MICROLITER": "μL",
    "MILLIGRAM_PER_LITER": "mg/L",
    "MILLIGRAM_PER_LITER_OF_BICARBONATE": "mg HCO3-/L",
    "MILLIGRAM_PER_LITER_OF_CHLORIDE": "mg Cl-/L",
    "MILLIGRAM_PER_LITER_OF_PHOSPHATE": "mg PO4-3/L",
    "MILLIGRAM_PER_LITER_OF_PHOSPHOR": "mg P/L",
    "MILLIGRAM_PER_LITER_OF_SODIUM_CHLORIDE": "mg NaCl/L",
    "MILLIGRAM_PER_LITER_OF_SULFATE": "mg SO4-2/L",
    "MILLILITER": "mL",
    "MMOLES_PER_LITER": "mmoles H+/L",
    "MOL_PER_LITER": "mol/L",
    "PERCENT": "%",
    "PERCENT_WEIGHT_VOLUME": "% p/v",
}


DEFAULT_TRANSFORM: Callable[..., Dict[str, Any]] = albumentations.Compose([
    albumentations.GlassBlur(max_delta=4, p=0.5),
    albumentations.ISONoise(intensity=(0.10, 0.50), always_apply=True, p=0.9),
    albumentations.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
])


T_co = TypeVar('T_co', covariant=True)


class SizedDataset(Sized, Dataset[T_co]):
    def __init__(self) -> None:
        super(SizedDataset, self).__init__()


class SampleDataset(SizedDataset[Sample]):
    def __init__(self, base_dirs: Union[str, Iterable[str]], *, progress_bar: bool = True, skip_blank_samples: bool = True, skip_incomplete_samples: bool = True, skip_inference_sample: bool = True, skip_training_sample: bool = False, verbose: bool = True) -> None:
        super(SampleDataset, self).__init__()
        # Load the list of samples.
        all_samples, blanks = self._load_samples(
            [base_dirs] if isinstance(base_dirs, str) else list(base_dirs),
            progress_bar=progress_bar,
            verbose=verbose,
        )
        # Try to assign the blank sample and skip incomplete samples, if required to.
        self._samples: List[Sample] = list()
        for sample in tqdm(all_samples, desc="Assigning blank samples", leave=False, disable=not progress_bar):
            if not os.path.isfile(sample["fileName"]) and skip_incomplete_samples:
                continue
            if sample["blankFileName"] is not None:
                if not os.path.isfile(sample["blankFileName"]) and skip_incomplete_samples:
                    continue
                sample["blank"] = blanks.get(sample["blankFileName"], None)
            if (sample["isBlankSample"] and not skip_blank_samples) or (sample["isInferenceSample"] and not skip_inference_sample) or (sample["isTrainingSample"] and not skip_training_sample):
                self._samples.append(sample)
        # Sort samples by date.
        #self._samples.sort(key=lambda sample: sample["datetime"])

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def _load_samples(self, base_dirs: List[str], *, progress_bar: bool, verbose: bool) -> Tuple[List[Sample], Dict[str, Sample]]:
        samples: List[Sample] = list()
        blanks: Dict[str, Sample] = dict()  # Dict[filename, sample]
        dirs = sorted(base_dirs)
        with tqdm(desc="Scanning folders", total=len(dirs), leave=False, disable=not progress_bar) as pbar_dirs:
            while len(dirs) != 0:
                dirname = dirs.pop(0)
                for basename in tqdm(sorted(os.listdir(dirname)), desc="Parsing JSONs", leave=False, disable=not progress_bar):
                    path = os.path.join(dirname, basename)
                    if os.path.isfile(path) and basename.lower().endswith(".json"):
                        with open(path, "r", encoding="utf8") as file:
                            sample = self._parse_sample(json.load(file), record_path=path, verbose=verbose)
                            samples.append(sample)
                            if sample["isBlankSample"]:
                                blanks[sample["fileName"]] = sample
                    elif os.path.isdir(path):
                        dirs.append(path)
                        pbar_dirs.total += 1
                        pbar_dirs.refresh()
                pbar_dirs.update(1)
        return samples, blanks

    def _parse_auxiliary_solution(self, type: str, raw_solution: Dict[str, Any]) -> AuxiliarySolution:
        return AuxiliarySolution(
            type=type,
            name=raw_solution["name"].strip(),
            components=[self._parse_auxiliary_solution_component(raw_component) for raw_component in raw_solution["components"]]
        )

    def _parse_auxiliary_solution_component(self, raw_component: Dict[str, Any]) -> AuxiliarySolutionComponent:
        return AuxiliarySolutionComponent(
            name=raw_component["name"].strip(),
            concentration=raw_component["concentration"],
            concentrationUnit=UNITS[raw_component.get("concentrationUnit", CONCENTRATION_UNIT_FROM_FUNCTION[raw_component["function"]])],
            function=SOLUTION_COMPONENT_FUNCTIONS[raw_component["function"]],
            batch=raw_component["batch"].strip()
        )

    def _parse_sample(self, raw_record: Dict[str, Any], record_path: str, *, verbose: bool) -> Sample:
        dirname, _ = os.path.split(record_path)
        # Check the extra image files.
        extra_filenames: List[str] = list()
        for extraFileName in raw_record["sample"].get("extraFileNames", []):
            extra_filename = os.path.join(dirname, extraFileName)
            if os.path.isfile(extra_filename):
                extra_filenames.append(extra_filename)
            else:
                warnings.warn(f'The extra sample image "{extra_filename}" is missing.') if verbose else None
        # Check the main image file.
        filename = os.path.join(dirname, raw_record["sample"]["fileName"])
        if not os.path.isfile(filename):
            if len(extra_filenames) > 0:
                warnings.warn(f'The sample image "{filename}" is missing. It was replaced by one of the extra images.') if verbose else None
                filename = extra_filenames.pop()
            else:
                warnings.warn(f'The sample image "{filename}" is missing and could not be replaced by one of the extra images.') if verbose else None
        # Check the blank image file.
        blank_filename = os.path.join(dirname, raw_record["sample"]["blankFileName"]) if raw_record["sample"]["blankFileName"] is not None else None
        if blank_filename is not None and not os.path.isfile(blank_filename):
            warnings.warn(f'The blank sample image "{blank_filename}" is missing.') if verbose else None
        is_blank_sample = blank_filename is None
        # Call methods implemented by the subclass to parse specialized data.
        auxiliary_solutions = self._parse_auxiliary_solutions(raw_record["sample"])
        stock_value, estimated_value, value_unit = self._parse_values(raw_record["sample"])
        value_unit = UNITS[value_unit]
        if stock_value is None and estimated_value is None:
            estimated_value = float("NaN")  # It is an inferece sample, but the app could not estimate the concentration value.
        # Parse source.
        source_stock = Stock(
            name=raw_record["sample"]["sourceStock"]["name"].strip(),
            components=[self._parse_solution_component(raw_component) for raw_component in raw_record["sample"]["sourceStock"]["components"]],
            value=stock_value,
            valueUnit=value_unit,
            aliquots=[self._parse_stock_aliquot(raw_aliquot) for raw_aliquot in raw_record["sample"]["sourceStock"]["aliquots"]],
        )
        source_aliquot = self._parse_stock_aliquot(raw_record["sample"]["sourceAliquot"])
        # Set some useful variables.
        stock_factor = raw_record["sample"]["stockFactor"]
        standard_volume = raw_record["sample"]["standardVolume"]
        used_volume = raw_record["sample"]["usedVolume"]
        volume_unit = UNITS[raw_record["sample"]["volumeUnit"]]
        theoretical_value = compute_theoretical_value(stock_value, source_aliquot["aliquot"], source_aliquot["finalVolume"]) if stock_value is not None else None
        corrected_theoretical_value = correct_theoretical_value(theoretical_value, standard_volume, used_volume, stock_factor) if theoretical_value is not None else None
        is_inference_sample = not (is_blank_sample or estimated_value is None)
        is_training_sample = not (is_blank_sample or theoretical_value is None)
        name = source_aliquot["name"].strip() if (is_blank_sample or is_inference_sample) else f'{source_stock["name"].strip()} - {source_aliquot["name"].strip()}'
        # Create the sample.
        return Sample(
            # Properties filled by this parser using data stored in the raw record.
            app=App(
                packageName=raw_record["app"]["packageName"],
                appName=raw_record["app"]["appName"],
                versionName=raw_record["app"]["versionName"],
            ),
            device=Device(
                model=raw_record["device"]["model"],
                manufacturer=raw_record["device"]["manufacturer"],
                androidVersion=raw_record["device"]["androidVersion"],
            ),
            sourceStock=source_stock,
            sourceAliquot=source_aliquot,
            stockFactor=stock_factor,
            standardVolume=standard_volume,
            usedVolume=used_volume,
            volumeUnit=volume_unit,
            chamberType=ChamberType[raw_record["sample"].get("chamberType", "CUVETTE")],
            fileName=filename,
            extraFileNames=extra_filenames,
            blankFileName=blank_filename,
            analystName=raw_record["sample"]["analystName"].strip(),
            notes=raw_record["sample"]["notes"].strip(),
            datetime=datetime.strptime(raw_record["sample"]["datetime"], DATETIME_FORMAT),
            # Properties computed on the fly by this parser.
            recordPath=record_path,
            name=name,
            isBlankSample=is_blank_sample,
            isInferenceSample=is_inference_sample,
            isTrainingSample=is_training_sample,
            theoreticalValue=theoretical_value,
            correctedTheoreticalValue=corrected_theoretical_value,
            blank=None,  # To be assigned next in the class constructor.
            # Properties computed on the fly by the parsers implemented by the subclass.
            auxiliarySolutions=auxiliary_solutions,
            estimatedValue=estimated_value,
            valueUnit=value_unit,
            # Properties set by the data augmentation module and other modules.
            referenceSample=None,
            extra=None,
        )

    def _parse_solution_component(self, raw_component: Dict[str, Any]) -> SolutionComponent:
        return SolutionComponent(
            name=raw_component["name"].strip(),
            concentration=raw_component["concentration"],
            concentrationUnit=UNITS[raw_component["concentrationUnit"]],
            batch=raw_component["batch"].strip()
        )

    def _parse_stock_aliquot(self, raw_aliquot: Dict[str, Any]) -> StockAliquot:
        return StockAliquot(
            name=raw_aliquot["name"].strip(),
            finalVolume=raw_aliquot["finalVolume"],
            finalVolumeUnit=UNITS[raw_aliquot["finalVolumeUnit"]],
            aliquot=raw_aliquot["aliquot"],
            aliquotUnit=UNITS[raw_aliquot["aliquotUnit"]],
        )

    @abstractmethod
    def _parse_auxiliary_solutions(self, raw_sample: Dict[str, Any]) -> List[AuxiliarySolution]:
        raise NotImplementedError  # To be implemented by the subclass.

    @abstractmethod
    def _parse_values(self, raw_sample: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        raise NotImplementedError  # To be implemented by the subclass.


class ExpandedSampleDataset(SizedDataset[Sample]):
    def __init__(self, dataset: Dataset[Sample], *, progress_bar: bool = True) -> None:
        super(ExpandedSampleDataset, self).__init__()
        # Expand blank samples.
        expanded_blanks: Dict[int, List[Sample]] = dict()  # Dict[id, List[expanded_blank]]
        for sample in tqdm(dataset, desc="Expanding blanks", leave=False, disable=not progress_bar): # type: ignore
            blank = sample if sample["isBlankSample"] else sample["blank"]
            if blank is not None:
                if id(blank) not in expanded_blanks:
                    expanded_blank: List[Sample] = list()
                    for filename, name_appendix in zip([blank["fileName"], *blank["extraFileNames"]], ["", *map(lambda ind: f' [Extra {ind}]', range(1, len(blank["extraFileNames"]) + 1))]): # type: ignore
                        expanded_blank.append(self._create_sample_from(blank, name=f'{blank["name"]}{name_appendix}', fileName=filename))
                    expanded_blanks[id(blank)] = expanded_blank
        # Create the expanded set of regular samples by combining the main and extra images of the regular and blank samples.
        self._samples: List[Sample] = list()
        for sample in tqdm(dataset, desc="Expanding regular samples", leave=False, disable=not progress_bar): # type: ignore
            if sample["isBlankSample"]:
                self._samples.extend(expanded_blanks[id(sample)])
            elif sample["blank"] is not None:
                for filename, name_appendix in zip([sample["fileName"], *sample["extraFileNames"]], ["", *map(lambda ind: f' [Extra {ind}]', range(1, len(sample["extraFileNames"]) + 1))]): # type: ignore
                    for blank in expanded_blanks[id(sample["blank"])]:
                        self._samples.append(self._create_sample_from(sample, name=f'{sample["name"]}{name_appendix}', fileName=filename, blankFileName=blank["fileName"], blank=blank))
            else:
                for filename, name_appendix in zip([sample["fileName"], *sample["extraFileNames"]], ["", *map(lambda ind: f' [Extra {ind}]', range(1, len(sample["extraFileNames"]) + 1))]): # type: ignore
                    self._samples.append(self._create_sample_from(sample, name=f'{sample["name"]}{name_appendix}', fileName=filename, blankFileName=sample["blankFileName"]))

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def _create_sample_from(self, sample: Sample, **kwargs: Any) -> Sample:
        new_sample = copy.deepcopy(sample)
        new_sample["extraFileNames"] = []
        new_sample["recordPath"] = None
        new_sample["referenceSample"] = sample
        for key, value in kwargs.items():
            new_sample[key] = value
        return new_sample


class ProcessedSample:
    def __init__(self, sample: Sample, *, compute_masks_func: Callable[[np.ndarray, np.ndarray, ChamberType], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], compute_pmf_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]], postfix: str, root_dir: str, transform: Optional[Callable[..., Dict[str, Any]]]) -> None:
        # Define some properties and local functions.
        self._bgr_img_ext: Dict[str, str] = dict()
        def make_bgr_image(prefix: str, original_path: str) -> None:
            ext = os.path.splitext(original_path)[1] if transform is None else '.png'
            path = self._path(prefix, None, ext)
            if not os.path.isfile(path):
                if transform is None:
                    shutil.copyfile(original_path, path)
                else:
                    bgr_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
                    if bgr_img is None:
                        raise RuntimeError(f'Can\'t load the file "{original_path}"')
                    if not cv2.imwrite(path, transform(image=bgr_img)["image"]):
                        raise RuntimeError(f'Can\'t save the file "{path}"')
            self._bgr_img_ext[prefix] = ext
        # Set sample.
        self.sample: Final[Sample] = sample
        # Set functions to compute masks and PMFs.
        self._compute_masks = compute_masks_func
        self._compute_pmf = compute_pmf_func
        # Set root directory.
        self._root_dir = root_dir
        # Set sample's prefix and BGR images.
        self.sample_prefix: Final[str] = f'{os.path.splitext(os.path.basename(sample["fileName"]))[0]}{postfix}'
        # added to identify date and analyst name
        self.identifier: Final[str] = str(sample["datetime"].year)+' '+str(sample["datetime"].month)+' '+str(sample["datetime"].day)+' '+sample["analystName"]

        make_bgr_image(self.sample_prefix, sample["fileName"])
        # Set blank's prefix and BGR images.
        if sample["blankFileName"] is not None and os.path.isfile(sample["blankFileName"]):
            blank_prefix = f'{os.path.splitext(os.path.basename(sample["blankFileName"]))[0]}{postfix}'
            make_bgr_image(blank_prefix, sample["blankFileName"])
        else:
            blank_prefix = None
        self.blank_prefix: Final[Optional[str]] = blank_prefix

    def _bgr_image(self, prefix: str) -> np.ndarray:
        path = self._path(prefix, None, self._bgr_img_ext[prefix])
        bgr_img = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise RuntimeError(f'Can\'t load the file "{path}"')
        return bgr_img

    def _lab_image(self, prefix: str) -> np.ndarray:
        path = self._path(prefix, "lab_img", ".npz")
        if os.path.isfile(path):
            lab_img = next(iter(np.load(path).values()))
        else:
            lab_img = bgr_to_lab(self._bgr_image(prefix))
            np.savez_compressed(path, lab_img)
        return lab_img

    def _mask(self, prefix: str, key: str) -> np.ndarray:
        data: Dict[str, np.ndarray] = dict()
        path = self._path(prefix, key, ".npz")
        if os.path.isfile(path):
            data[key] = next(iter(np.load(path).values()))
        else:
            data["bright_msk"], data["grid_msk"], data["analyte_msk"], data["lab_white"] = self._compute_masks(self._bgr_image(prefix), self._lab_image(prefix), self.sample["chamberType"])
            for name, value in data.items():
                np.savez_compressed(self._path(prefix, name, ".npz"), value)
        return data[key]

    def _path(self, prefix: str, key: Optional[str], ext: str) -> str:
        return os.path.join(self._root_dir, f'{prefix}{f"_{key}" if key is not None else ""}{ext}')

    def _pmf(self, prefix: str, key: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        data: Dict[str, Any] = dict()
        path = self._path(prefix, key, ".npz")
        if os.path.isfile(path):
            if key == "pmf":
                data[key] = next(iter(np.load(path).values()))
            else:
                with np.load(path) as npz:
                    data[key] = (npz["img_ind"], npz["pmf_ind"])
        else:
            data["pmf"], data["img_to_pmf"] = self._compute_pmf(self._lab_image(prefix), self._mask(prefix, "analyte_msk"), self._mask(prefix, "lab_white"))
            np.savez_compressed(self._path(prefix, "pmf", ".npz"), data["pmf"])
            np.savez_compressed(self._path(prefix, "img_to_pmf", ".npz"), img_ind=data["img_to_pmf"][0], pmf_ind=data["img_to_pmf"][1])
        return data[key]


    @property
    def blank_analyte_mask(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._mask(self.blank_prefix, "analyte_msk")

    @property
    def blank_bgr_image(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._bgr_image(self.blank_prefix)

    @property
    def blank_bright_mask(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._mask(self.blank_prefix, "bright_msk")

    @property
    def blank_grid_mask(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._mask(self.blank_prefix, "grid_msk")

    @property
    def blank_image_to_pmf(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.blank_prefix is not None
        return self._pmf(self.blank_prefix, "img_to_pmf") # type: ignore

    @property
    def blank_lab_image(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._lab_image(self.blank_prefix)

    @property
    def blank_lab_white(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._mask(self.blank_prefix, "lab_white")

    @property
    def blank_pmf(self) -> np.ndarray:
        assert self.blank_prefix is not None
        return self._pmf(self.blank_prefix, "pmf") # type: ignore

    @property
    def calibrated_pmf(self) -> np.ndarray:
        path = self._path(f'{self.sample_prefix}-{self.blank_prefix}', "calibrated_pmf", ".npz")
        if os.path.isfile(path):
            calibrated_pmf = next(iter(np.load(path).values()))
        else:
            assert self.blank_prefix is not None
            calibrated_pmf = compute_calibrated_pmf(blank_pmf=self._pmf(self.blank_prefix, "pmf"), sample_pmf=self._pmf(self.sample_prefix, "pmf")) # type: ignore
            np.savez_compressed(path, calibrated_pmf)
        return calibrated_pmf

    @property
    def has_valid_blank(self) -> bool:
        return self.sample["blankFileName"] is not None and os.path.isfile(self.sample["blankFileName"])

    @property
    def sample_analyte_mask(self) -> np.ndarray:
        return self._mask(self.sample_prefix, "analyte_msk")

    @property
    def sample_bgr_image(self) -> np.ndarray:
        return self._bgr_image(self.sample_prefix)

    @property
    def sample_bright_mask(self) -> np.ndarray:
        return self._mask(self.sample_prefix, "bright_msk")

    @property
    def sample_grid_mask(self) -> np.ndarray:
        return self._mask(self.sample_prefix, "grid_msk")

    @property
    def sample_image_to_pmf(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._pmf(self.sample_prefix, "img_to_pmf") # type: ignore

    @property
    def sample_lab_image(self) -> np.ndarray:
        return self._lab_image(self.sample_prefix)

    @property
    def sample_lab_white(self) -> np.ndarray:
        return self._mask(self.sample_prefix, "lab_white")

    @property
    def sample_pmf(self) -> np.ndarray:
        return self._pmf(self.sample_prefix, "pmf") # type: ignore

    def sample_prefix(self) -> str:
        return self.sample_prefix


class ProcessedSampleDataset(SizedDataset[ProcessedSample]):
    def __init__(self, dataset: SizedDataset[Sample], cache_dir: str, *, num_augmented_samples: int = 0, progress_bar: bool = True, transform: Callable[..., Dict[str, Any]] = DEFAULT_TRANSFORM, **kwargs: Any) -> None:
        super(ProcessedSampleDataset, self).__init__()
        os.makedirs(cache_dir, exist_ok=True)
        # Copy original BGR images of samples to root_dir and augment them, if needed.
        self.samples = dataset
        self._processed_samples: List[ProcessedSample] = list()
        with tqdm(desc="Copying images to cache", total=len(dataset) * (num_augmented_samples + 1), leave=False, disable=not progress_bar) as pbar:
            for sample in dataset:
                self._processed_samples.append(ProcessedSample(
                    sample,
                    compute_masks_func=lambda bgr_img, lab_img, chamber_type: self._compute_masks(bgr_img, lab_img, chamber_type),
                    compute_pmf_func=lambda lab_img, analyte_msk, lab_white: self._compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white),
                    postfix="",
                    root_dir=cache_dir,
                    transform=None,
                ))
                pbar.update(1)
                # Create augmented versions of the current sample.
                for ind in range(1, num_augmented_samples + 1):
                    # Get the augmented images of the actual and blank samples.
                    self._processed_samples.append(ProcessedSample(
                        sample,
                        compute_masks_func=lambda bgr_img, lab_img, chamber_type: self._compute_masks(bgr_img, lab_img, chamber_type),
                        compute_pmf_func=lambda lab_img, analyte_msk, lab_white: self._compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white),
                        postfix=f'-augmented-{ind}',
                        root_dir=cache_dir,
                        transform=transform,
                    ))
                    pbar.update(1)

    def get_samples(self):
        return self.samples

    def __getitem__(self, index: int) -> ProcessedSample:
        return self._processed_samples[index]

    def __len__(self) -> int:
        return len(self._processed_samples)


    @abstractmethod
    def _compute_masks(self, bgr_img: np.ndarray, lab_img: np.ndarray, chamber_type: ChamberType) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError  # To be implemented by the subclass.

    @abstractmethod
    def _compute_pmf(self, lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError  # To be implemented by the subclass.

    def compute_calibrated_pmf_roi(self, reduction_level: float) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[float, float]]:
        min_prob, max_prob = 0.0, 0.0
        used_msk: Union[bool, np.ndarray] = False
        for processed_sample in self._processed_samples:
            if processed_sample.has_valid_blank:
                calibrated_pmf = processed_sample.calibrated_pmf
                sorted_probs = calibrated_pmf.flatten()
                sorted_probs.sort()
                min_prob = min(min_prob, sorted_probs[0].item())
                max_prob = max(max_prob, sorted_probs[-1].item())
                where = np.searchsorted(sorted_probs.cumsum(), reduction_level, side="left")
                used_msk |= calibrated_pmf > sorted_probs[where]
        bounds: List[Tuple[int, int]] = list()
        if isinstance(used_msk, np.ndarray):
            for axis in range(used_msk.ndim):
                used_axis = np.argwhere(np.logical_or.reduce(used_msk, axis=axis, initial=None))
                bounds.append((int(np.min(used_axis)), int(np.max(used_axis))))
        return tuple(bounds), (min_prob, max_prob)

    def compute_true_value_statistics(self) -> Dict[str, float]:
        values = np.asarray([processed_sample.sample["correctedTheoreticalValue"] for processed_sample in self._processed_samples if processed_sample.sample["correctedTheoreticalValue"] is not None], dtype=np.float32)
        median = np.median(values)
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(median),
            "mad": float(np.median(np.abs(values - median))),
        }
