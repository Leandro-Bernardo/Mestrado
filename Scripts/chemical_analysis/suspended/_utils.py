from .._utils import _compute_masks, lab_to_normalized
from ..typing import ChamberType
from typing import Optional, Tuple
import numpy as np


def compute_masks(bgr_img: np.ndarray, lab_img: Optional[np.ndarray], chamber_type: ChamberType) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    return _compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type, min_bright_threshould=255)


def compute_pmf(lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray, lab_mean: np.ndarray, lab_sorted_eigenvectors: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # Map L*a*b* coordinates to indices.
    ind = (255 * lab_to_normalized(lab_img[analyte_msk], lab_white=lab_white, lab_mean=lab_mean, lab_sorted_eigenvectors=lab_sorted_eigenvectors, out_channels=(0, 1))).astype(np.int64)
    in_range = np.logical_and(0 <= ind, ind < 256).all(axis=1)
    ind_in_range = ind[in_range, ...]
    # Compute the frequency of projected L*a*b* entries in the sample.
    pmf = np.zeros((256, 256), np.float32)
    np.add.at(pmf, (ind_in_range[:, 0], ind_in_range[:, 1]), 1)
    pmf /= pmf.sum()
    img_to_pmf = (np.stack(np.nonzero(analyte_msk), axis=1)[in_range, ...], ind_in_range)  #TODO Testar in_range
    # Return the PMF of the projected L*a*b* entries and extra data produced during the process.
    return pmf, img_to_pmf
