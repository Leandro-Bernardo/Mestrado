from .._dataset import ProcessedSampleDataset, SampleDataset
from ..typing import AuxiliarySolution, ChamberType
from ._utils import compute_masks, compute_pmf
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class IronOxidSampleDataset(SampleDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _parse_auxiliary_solutions(self, raw_sample: Dict[str, Any]) -> List[AuxiliarySolution]:
        return [
        ]

    def _parse_values(self, raw_sample: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        return (
            # theoretical_value = stock_value * (aliquot / final_volume)
            # iron_oxid_theoretical_value = estimatedSolidMass/(filteredVolume/1000)

             # correct_theoretical_value = theoretical_value * (used_volume (=1) / standard_volume (=1)) * stock_factor (=1)   #nao é preocupação

            #(raw_sample["sourceStock"]["estimatedSolidMass"])/(raw_sample["sourceStock"]["filteredVolume"]/1000),
            #(raw_sample.get("estimatedSolidMass", None))/(raw_sample.get("filteredVolume", None)/1000),
            raw_sample.get("estimatedSolidMass", None),
            ## outros analitos pegam analyteConcentration:  raw_sample["sourceStock"].get("iron3Concentration", None),  # =stock value
            raw_sample.get("estimatedConcentration", None),             # estimated value (=None)
            raw_sample["sourceStock"].get("suspendedConcentrationUnit", "PARTS_PER_MILLION"), #value unit
        )


class ProcessedIronOxidSampleDataset(ProcessedSampleDataset):
    def __init__(self, *args: Any, lab_mean: np.ndarray, lab_sorted_eigenvectors: np.ndarray, **kwargs: Any) -> None:
        self._lab_mean = lab_mean
        self._lab_sorted_eigenvectors = lab_sorted_eigenvectors
        super().__init__(*args, **kwargs)

    def _compute_masks(self, bgr_img: np.ndarray, lab_img: np.ndarray, chamber_type: ChamberType) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (bright_msk, grid_msk, analyte_msk), _, lab_white = compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type)
        return bright_msk, grid_msk, analyte_msk, lab_white

    def _compute_pmf(self, lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._lab_mean, lab_sorted_eigenvectors=self._lab_sorted_eigenvectors)
