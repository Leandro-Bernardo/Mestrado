from .._dataset import ProcessedSampleDataset, SampleDataset
from ..typing import AuxiliarySolution, ChamberType
from ._utils import compute_masks, compute_pmf
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Iron3SampleDataset(SampleDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _parse_auxiliary_solutions(self, raw_sample: Dict[str, Any]) -> List[AuxiliarySolution]:
        return [
            self._parse_auxiliary_solution("Complexant", raw_sample["complexant"]),
            self._parse_auxiliary_solution("Acid", raw_sample["acid"]),
        ]

    def _parse_values(self, raw_sample: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        return (
            raw_sample["sourceStock"].get("iron3Concentration", None),
            raw_sample.get("estimatedConcentration", None),
            raw_sample["sourceStock"].get("iron3ConcentrationUnit", "MILLIGRAM_PER_LITER_OF_IRON3"),
        )


class ProcessedIron3SampleDataset(ProcessedSampleDataset):
    def __init__(self, *args: Any, lab_mean: np.ndarray, lab_sorted_eigenvectors: np.ndarray, **kwargs: Any) -> None:
        self._lab_mean = lab_mean
        self._lab_sorted_eigenvectors = lab_sorted_eigenvectors
        super().__init__(*args, **kwargs)

    def _compute_masks(self, bgr_img: np.ndarray, lab_img: np.ndarray, chamber_type: ChamberType) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (bright_msk, grid_msk, analyte_msk), _, lab_white = compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type)
        return bright_msk, grid_msk, analyte_msk, lab_white

    def _compute_pmf(self, lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._lab_mean, lab_sorted_eigenvectors=self._lab_sorted_eigenvectors)
