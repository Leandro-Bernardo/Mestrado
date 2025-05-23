from .._dataset import ProcessedSampleDataset, SampleDataset
from ..typing import AuxiliarySolution, ChamberType
from ._utils import compute_masks, compute_pmf
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class PhSampleDataset(SampleDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _parse_auxiliary_solutions(self, raw_sample: Dict[str, Any]) -> List[AuxiliarySolution]:
        return [ 
            self._parse_auxiliary_solution("phIndicatorMix", raw_sample["phIndicatorMix"]),
        ]
    
    def _parse_values(self, raw_sample: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        return (
            raw_sample["sourceStock"].get("pHValue", None),
            raw_sample.get("estimatedConcentration", None),
            raw_sample["sourceStock"].get("pHValueUnit", "POWER_OF_HYDROGEN"),
        )


class ProcessedPhSampleDataset(ProcessedSampleDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ph_values = self.get_samples()

    @property
    def analyte_values(self):
        return self._ph_values

    def _compute_masks(self, bgr_img: np.ndarray, lab_img: np.ndarray, chamber_type: ChamberType) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (bright_msk, grid_msk, analyte_msk), _, lab_white = compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type)
        return bright_msk, grid_msk, analyte_msk, lab_white

    def _compute_pmf(self, lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
