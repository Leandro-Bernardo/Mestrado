from .._utils import LAB_CIE_D65, _compute_masks
from ..typing import ChamberType
from typing import Optional, Tuple
import numpy as np


def compute_masks(bgr_img: np.ndarray, lab_img: Optional[np.ndarray], chamber_type: ChamberType) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    return _compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type, min_bright_threshould=None)


def compute_pmf(lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    #TODO Utilizar lab_to_normalized
    # Map a*b* coordinates of pixels to the CIE standard illuminant D65 and convert the result to indices.
    ab_ind = np.flip(lab_img[analyte_msk, 1:] - (lab_white[1:] - LAB_CIE_D65[1:] - 128), 1).astype(np.int64)
    in_range = np.logical_and(np.logical_and(0 <= ab_ind[:, 0], ab_ind[:, 0] < 256), np.logical_and(0 <= ab_ind[:, 1], ab_ind[:, 1] < 256))
    ab_ind = ab_ind[in_range, ...]
    # Compute the frequency of each a*b* pair in the sample.
    pmf = np.zeros((256, 256), np.float32)
    np.add.at(pmf, (ab_ind[:, 0], ab_ind[:, 1]), 1)
    pmf /= pmf.sum()
    img_to_pmf = (np.stack(np.nonzero(analyte_msk), axis=1)[in_range, ...], ab_ind)
    # Return the PMF of the a*b* pairs in the image of the sample and extra data produced during the process.
    return pmf, img_to_pmf
