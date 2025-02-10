from . import alkalinity, bisulfite2d, chloride, iron2, iron3, ph, phosphate, sulfate
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
BISULFITE_IMPRECISION_WARNING: Final[ErrorCode] = 0x20
BISULFITE_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x201
BISULFITE_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x202
CHLORIDE_IMPRECISION_WARNING: Final[ErrorCode] = 0x30
CHLORIDE_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x301
CHLORIDE_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x302
EMULSION_IMPRECISION_WARNING: Final[ErrorCode] = 0x40
EMULSION_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x401
EMULSION_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x402
IRON2_IMPRECISION_WARNING: Final[ErrorCode] = 0x50
IRON2_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x501
IRON2_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x502
IRON3_IMPRECISION_WARNING: Final[ErrorCode] = 0x60
IRON3_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x601
IRON3_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x602
PH_IMPRECISION_WARNING: Final[ErrorCode] = 0x70
PH_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x701
PH_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x702
PHOSPHATE_IMPRECISION_WARNING: Final[ErrorCode] = 0x80
PHOSPHATE_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x801
PHOSPHATE_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x802
SULFATE_IMPRECISION_WARNING: Final[ErrorCode] = 0x90
SULFATE_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x901
SULFATE_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x902
SUSPENDED_IMPRECISION_WARNING: Final[ErrorCode] = 0x100
SUSPENDED_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x1001
SUSPENDED_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x1002
REDOX_IMPRECISION_WARNING: Final[ErrorCode] = 0x110
REDOX_LOWER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x1101
REDOX_UPPER_BOUND_IMPRECISION_WARNING: Final[ErrorCode] = 0x1102


ALKALINITY_BLANK_VALIDITY = timedelta(hours=8)
BISULFITE_BLANK_VALIDITY = timedelta(hours=8)
CHLORIDE_BLANK_VALIDITY = timedelta(hours=8)
EMULSION_BLANK_VALIDITY = timedelta(hours=8)
IRON2_BLANK_VALIDITY = timedelta(hours=8)
IRON3_BLANK_VALIDITY = timedelta(hours=8)
PH_BLANK_VALIDITY = timedelta(hours=8)
PHOSPHATE_BLANK_VALIDITY = timedelta(hours=8)
SULFATE_BLANK_VALIDITY = timedelta(hours=8)
SUSPENDED_BLANK_VALIDITY = timedelta(hours=8)
REDOX_BLANK_VALIDITY = timedelta(hours=8)


class Blank(NamedTuple):
    data: Optional[np.ndarray]
    time: datetime


class AnalysisFacade:
    def __init__(self) -> None:
        # Get device.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set blank samples.
        self._alkalinity_blank = Blank(None, datetime.now())
        self._bisulfite_blank = Blank(None, datetime.now())
        self._chloride_blank = Blank(None, datetime.now())
        self._emulsion_blank = Blank(None, datetime.now())
        self._iron2_blank = Blank(None, datetime.now())
        self._iron3_blank = Blank(None, datetime.now())
        self._ph_blank = Blank(None, datetime.now())
        self._phosphate_blank = Blank(None, datetime.now())
        self._sulfate_blank = Blank(None, datetime.now())
        self._suspension_blank = Blank(None, datetime.now())
        self._redox_blank = Blank(None, datetime.now())
        # Load checkpoint.
        self._alkalinity_net = alkalinity.AlkalinityNetwork.load_from_checkpoint(alkalinity.NETWORK_CHECKPOINT).to(self._device)
        self._alkalinity_net.eval()
        self._bisulfite_net = bisulfite2d.Bisulfite2DNetwork.load_from_checkpoint(bisulfite2d.NETWORK_CHECKPOINT).to(self._device)
        self._bisulfite_net.eval()
        self._chloride_net = chloride.ChlorideNetwork.load_from_checkpoint(chloride.NETWORK_CHECKPOINT).to(self._device)
        self._chloride_net.eval()
        self._iron2_net = iron2.Iron2Network.load_from_checkpoint(iron2.NETWORK_CHECKPOINT).to(self._device)
        self._iron2_net.eval()
        self._ph_net = ph.PhNetwork.load_from_checkpoint(ph.NETWORK_CHECKPOINT).to(self._device)
        self._ph_net.eval()
        self._iron3_net = iron3.Iron3Network.load_from_checkpoint(iron3.NETWORK_CHECKPOINT).to(self._device)
        self._iron3_net.eval()
        self._phosphate_net = phosphate.PhosphateNetwork.load_from_checkpoint(phosphate.NETWORK_CHECKPOINT).to(self._device)
        self._phosphate_net.eval()
        self._sulfate_net = sulfate.SulfateNetwork.load_from_checkpoint(sulfate.NETWORK_CHECKPOINT).to(self._device)
        self._sulfate_net.eval()
        # Load PCA statistics.
        stats = np.load(bisulfite2d.PCA_STATS)
        self._bisulfite_lab_mean = stats["lab_mean"]
        self._bisulfite_lab_sorted_eigenvectors = stats["lab_sorted_eigenvectors"]
        stats = np.load(chloride.PCA_STATS)
        self._chloride_lab_mean = stats["lab_mean"]
        self._chloride_lab_sorted_eigenvectors = stats["lab_sorted_eigenvectors"]
        stats = np.load(iron3.PCA_STATS)
        self._iron3_lab_mean = stats["lab_mean"]
        self._iron3_lab_sorted_eigenvectors = stats["lab_sorted_eigenvectors"]
        stats = np.load(iron2.PCA_STATS)
        self._iron2_lab_mean = stats["lab_mean"]
        self._iron2_lab_sorted_eigenvectors = stats["lab_sorted_eigenvectors"]
        stats = np.load(phosphate.PCA_STATS)
        self._phosphate_lab_mean = stats["lab_mean"]
        self._phosphate_lab_sorted_eigenvectors = stats["lab_sorted_eigenvectors"]
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

    def check_bisulfite_blank(self) -> ErrorCode:
        return self._check_blank(self._bisulfite_blank, BISULFITE_BLANK_VALIDITY)
    
    def check_chloride_blank(self) -> ErrorCode:
        return self._check_blank(self._chloride_blank, CHLORIDE_BLANK_VALIDITY)
    
    def check_emulsion_blank(self) -> ErrorCode:
        return self._check_blank(self._emulsion_blank, EMULSION_BLANK_VALIDITY)
    
    def check_iron2_blank(self) -> ErrorCode:
        return self._check_blank(self._iron2_blank, IRON2_BLANK_VALIDITY)
    
    def check_iron3_blank(self) -> ErrorCode:
        return self._check_blank(self._iron3_blank, IRON3_BLANK_VALIDITY)
    
    def check_ph_blank(self) -> ErrorCode:
        return self._check_blank(self._ph_blank, PH_BLANK_VALIDITY)
    
    def check_phosphate_blank(self) -> ErrorCode:
        return self._check_blank(self._phosphate_blank, PHOSPHATE_BLANK_VALIDITY)

    def check_sulfate_blank(self) -> ErrorCode:
        return self._check_blank(self._sulfate_blank, SULFATE_BLANK_VALIDITY)
    
    def check_suspended_blank(self) -> ErrorCode:
        return self._check_blank(self._suspended_blank, SUSPENDED_BLANK_VALIDITY)
    
    def check_redox_blank(self) -> ErrorCode:
        return self._check_blank(self._redox_blank, REDOX_BLANK_VALIDITY)

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

    def estimate_bisulfite_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_bisulfite_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._bisulfite_blank
        assert blank_pmf is not None
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        (_, _, analyte_msk), lab_img, lab_white = bisulfite2d.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = bisulfite2d.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._bisulfite_lab_mean, lab_sorted_eigenvectors=self._bisulfite_lab_sorted_eigenvectors)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._bisulfite_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1, bounds[1][0]:bounds[1][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the concentration and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._bisulfite_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._bisulfite_net.expected_range
        if value <= lower:
            error_code = BISULFITE_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = BISULFITE_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)

    def estimate_chloride_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_chloride_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._chloride_blank
        assert blank_pmf is not None
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        (_, _, analyte_msk), lab_img, lab_white = chloride.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = chloride.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._chloride_lab_mean, lab_sorted_eigenvectors=self._chloride_lab_sorted_eigenvectors)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._chloride_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1, bounds[1][0]:bounds[1][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the concentration and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._chloride_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._chloride_net.expected_range
        if value <= lower:
            error_code = CHLORIDE_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = CHLORIDE_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)
    
    def estimate_iron3_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_iron3_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._iron3_blank
        assert blank_pmf is not None
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        (_, _, analyte_msk), lab_img, lab_white = iron3.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = iron3.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._iron3_lab_mean, lab_sorted_eigenvectors=self._iron3_lab_sorted_eigenvectors)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._iron3_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the concentration and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._iron3_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._iron3_net.expected_range
        if value <= lower:
            error_code = IRON3_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = IRON3_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)
    
    def estimate_iron2_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_iron2_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._iron2_blank
        assert blank_pmf is not None
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        (_, _, analyte_msk), lab_img, lab_white = iron2.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = iron2.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._iron2_lab_mean, lab_sorted_eigenvectors=self._iron2_lab_sorted_eigenvectors)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._iron2_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the concentration and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._iron2_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._iron2_net.expected_range
        if value <= lower:
            error_code = IRON2_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = IRON2_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)
    
    def estimate_ph_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_ph_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._ph_blank
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        assert blank_pmf is not None
        (_, _, analyte_msk), lab_img, lab_white = ph.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = ph.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._ph_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1, bounds[1][0]:bounds[1][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the ph and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._ph_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._ph_net.expected_range
        if value <= lower:
            error_code = PH_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = PH_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)

    def estimate_phosphate_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_phosphate_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._phosphate_blank
        assert blank_pmf is not None
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        (_, _, analyte_msk), lab_img, lab_white = phosphate.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = phosphate.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._phosphate_lab_mean, lab_sorted_eigenvectors=self._phosphate_lab_sorted_eigenvectors)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._phosphate_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the concentration and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._phosphate_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._phosphate_net.expected_range
        if value <= lower:
            error_code = PHOSPHATE_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = PHOSPHATE_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)

    def estimate_sulfate_concentration(self, sample_path: str, standard_volume: float, used_volume: float) -> Tuple[ErrorCode, float, float]:
        # Check blank sample integrity.
        error_code = self.check_sulfate_blank()
        if error_code != NO_ERROR:
            return error_code, float("NaN"), float("NaN")
        # Compute the calibrated PMF for the sample.
        blank_pmf, _ = self._sulfate_blank
        assert blank_pmf is not None
        sample_img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
        (_, _, analyte_msk), lab_img, lab_white = sulfate.compute_masks(bgr_img=sample_img, lab_img=None, chamber_type=ChamberType.POT)
        sample_pmf, _ = sulfate.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        calibrated_pmf = compute_calibrated_pmf(blank_pmf=blank_pmf, sample_pmf=sample_pmf)
        # Check whether the sample can be processed.
        bounds = self._sulfate_net.input_roi
        roi = calibrated_pmf[bounds[0][0]:bounds[0][1]+1]
        if roi.sum() <= 0.90:
            return REDUCTION_REQUIRED_ERROR, float("NaN"), float("NaN")  # 10% of the pixels samples doesn't fall on the ROI. Reduction is required.
        # Estimate the concentration and apply correction due to reduction.
        with torch.no_grad():
            value, _ = self._sulfate_net(torch.as_tensor(calibrated_pmf, dtype=torch.float32, device=self._device).unsqueeze(0))
        corrected_value = correct_predicted_value(value, standard_volume, used_volume)
        if math.isnan(value) or value < 0.0:
            return ESTIMATION_ERROR, float("NaN"), float("NaN")
        # Return the estimated and the corrected values.
        lower, upper = self._sulfate_net.expected_range
        if value <= lower:
            error_code = SULFATE_LOWER_BOUND_IMPRECISION_WARNING
        elif value >= upper:
            error_code = SULFATE_UPPER_BOUND_IMPRECISION_WARNING
        return error_code, float(value), float(corrected_value)

    def forget_alkalinity_blank(self) -> None:
        self.set_alkalinity_blank(None)

    def forget_bisulfite_blank(self) -> None:
        self.set_bisulfite_blank(None)

    def forget_chloride_blank(self) -> None:
        self.set_chloride_blank(None)
        
    def forget_emulsion_blank(self) -> None:
        self.set_emulsion_blank(None)
        
    def forget_iron2_blank(self) -> None:
        self.set_iron2_blank(None)

    def forget_iron3_blank(self) -> None:
        self.set_iron3_blank(None)

    def forget_ph_blank(self) -> None:
        self.set_ph_blank(None)

    def forget_phosphate_blank(self) -> None:
        self.set_phosphate_blank(None)

    def forget_sulfate_blank(self) -> None:
        self.set_sulfate_blank(None)
        
    def forget_suspended_blank(self) -> None:
        self.set_suspended_blank(None)
        
    def forget_redox_blank(self) -> None:
        self.set_redox_blank(None)

    def get_alkalinity_network_version(self) -> str:
        return self._get_version(self._alkalinity_net)

    def get_alkalinity_range(self) -> Tuple[float, float]:
        return self._alkalinity_net.expected_range

    def get_bisulfite_network_version(self) -> str:
        return self._get_version(self._bisulfite_net)

    def get_bisulfite_range(self) -> Tuple[float, float]:
        return self._bisulfite_net.expected_range

    def get_chloride_network_version(self) -> str:
        return self._get_version(self._chloride_net)

    def get_chloride_range(self) -> Tuple[float, float]:
        return self._chloride_net.expected_range
    
    def get_iron3_network_version(self) -> str:
        return self._get_version(self._iron3_net)
    
    def get_iron3_range(self) -> Tuple[float, float]:
        return self._iron3_net.expected_range
    
    
    def get_iron2_network_version(self) -> str:
        return self._get_version(self._iron2_net)
    
    def get_iron2_range(self) -> Tuple[float, float]:
        return self._iron2_net.expected_range
    
    def get_ph_network_version(self) -> str:
        return self._get_version(self._ph_net)

    def get_ph_range(self) -> Tuple[float, float]:
        return self._ph_net.expected_range

    def get_phosphate_network_version(self) -> str:
        return self._get_version(self._phosphate_net)

    def get_phosphate_range(self) -> Tuple[float, float]:
        return self._phosphate_net.expected_range

    def get_sulfate_network_version(self) -> str:
        return self._get_version(self._sulfate_net)

    def get_sulfate_range(self) -> Tuple[float, float]:
        return self._sulfate_net.expected_range

    def set_alkalinity_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = alkalinity.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = alkalinity.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        self._alkalinity_blank = Blank(blank_pmf, datetime.now())
    
    def set_bisulfite_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = bisulfite2d.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = bisulfite2d.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._bisulfite_lab_mean, lab_sorted_eigenvectors=self._bisulfite_lab_sorted_eigenvectors)
        self._bisulfite_blank = Blank(blank_pmf, datetime.now())
    
    def set_chloride_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = chloride.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = chloride.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._chloride_lab_mean, lab_sorted_eigenvectors=self._chloride_lab_sorted_eigenvectors)
        self._chloride_blank = Blank(blank_pmf, datetime.now())
    
    def set_emulsion_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_pmf = np.zeros((1, 1), dtype=np.float32)
        self._emulsion_blank = Blank(blank_pmf, datetime.now())
    
    def set_iron2_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = iron2.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = iron2.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._iron2_lab_mean, lab_sorted_eigenvectors=self._iron2_lab_sorted_eigenvectors)
        self._iron2_blank = Blank(blank_pmf, datetime.now())

    def set_iron3_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = iron3.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = iron3.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._iron3_lab_mean, lab_sorted_eigenvectors=self._iron3_lab_sorted_eigenvectors)
        self._iron3_blank = Blank(blank_pmf, datetime.now())

    
    def set_ph_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = ph.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = ph.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        self._ph_blank = Blank(blank_pmf, datetime.now())

    def set_phosphate_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = phosphate.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = phosphate.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white, lab_mean=self._phosphate_lab_mean, lab_sorted_eigenvectors=self._phosphate_lab_sorted_eigenvectors)
        self._phosphate_blank = Blank(blank_pmf, datetime.now())

    def set_sulfate_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_img = cv2.imread(blank_path, cv2.IMREAD_COLOR)
            (_, _, analyte_msk), lab_img, lab_white = sulfate.compute_masks(bgr_img=blank_img, lab_img=None, chamber_type=ChamberType.POT)
            blank_pmf, _ = sulfate.compute_pmf(lab_img=lab_img, analyte_msk=analyte_msk, lab_white=lab_white)
        self._sulfate_blank = Blank(blank_pmf, datetime.now())
    
    def set_suspended_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_pmf = np.zeros((1, 1), dtype=np.float32)
        self._suspended_blank = Blank(blank_pmf, datetime.now())
    
    def set_redox_blank(self, blank_path: Optional[str]) -> None:
        if blank_path is None:
            blank_pmf = None
        else:
            blank_pmf = np.zeros((1, 1), dtype=np.float32)
        self._redox_blank = Blank(blank_pmf, datetime.now())
