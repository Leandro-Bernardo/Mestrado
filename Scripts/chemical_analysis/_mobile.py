from . import alkalinity
from ._default import WHITEBALANCE_STATS
from ._model import Network
from ._utils import compute_calibrated_pmf, correct_predicted_value
from .typing import ChamberType
from datetime import datetime, timedelta
from typing import Final, NamedTuple, Optional, Tuple
import cv2
import numpy as np
import math
import torch


ErrorCode = int


NO_ERROR: Final[ErrorCode] = 0x000
BLANK_REQUIRED_ERROR: Final[ErrorCode] = 0x001
ESTIMATION_ERROR: Final[ErrorCode] = 0x002
FRESH_BLANK_REQUIRED_ERROR: Final[ErrorCode] = 0x003
REDUCTION_REQUIRED_ERROR: Final[ErrorCode] = 0x004
NOT_IMPLEMENTED_ERROR: Final[ErrorCode] = 0x0FF


ALKALINITY_IMPRECISION_WARNING: Final[ErrorCode] = 0x10
ALKALINITY_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x101
ALKALINITY_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x102


ALKALINITY_BLANK_VALIDITY = timedelta(hours=8)

class Blank(NamedTuple):
    data: Optional[np.ndarray]
    time: datetime


class AnalysisFacade:
    def __init__(self) -> None:
        # Get device.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set blank samples.
        self._alkalinity_blank = Blank(None, datetime.now())
        
        # Load checkpoint.
        self._alkalinity_net = alkalinity.AlkalinityNetwork.load_from_checkpoint(alkalinity.NETWORK_CHECKPOINT).to(self._device)
        self._alkalinity_net.eval()
       
        # Load white balance statistics.
        self._whitebalance_stats = dict(np.load(WHITEBALANCE_STATS))

    def _check_blank(self, blank: Blank, validity: timedelta) -> ErrorCode:
        blank_data, blank_time = blank
        if blank_data is None:
            return BLANK_REQUIRED_ERROR
        elif (datetime.now() - blank_time) > validity:
            return FRESH_BLANK_REQUIRED_ERROR
        return NO_ERROR

    def _get_version(self, net: Network) -> str:
        return net.version if hasattr(net, "version") else f"{net.__class__.__name__}-UnknownVersion"

    def check_alkalinity_blank(self) -> ErrorCode:
        return self._check_blank(self._alkalinity_blank, ALKALINITY_BLANK_VALIDITY)

    def estimate_alkalinity(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_alkalinity_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._alkalinity_blank
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        assert blank_pmf is not None
        (_, _, analyte_msk), lab_img, lab_white = alkalinity.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = alkalinity.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._alkalinity_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1, bounds[1][0]:bounds[1][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the alkalinity and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._alkalinity_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._alkalinity_net.expected_range
        if value <= lower:
            error_code = ALKALINITY_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = ALKALINITY_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)

   
    def forget_alkalinity_blank(self) -> None:
        self.set_alkalinity_blank(None)


    def get_alkalinity_network_version(self) -> str:
        return self._get_version(self._alkalinity_net)

    def get_alkalinity_range(self) -> Tuple[float, float]:
        return self._alkalinity_net.expected_range

    

    def set_alkalinity_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = alkalinity.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = alkalinity.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        self._alkalinity_blank = Blank(blank_pmf, datetime.now())
    
    