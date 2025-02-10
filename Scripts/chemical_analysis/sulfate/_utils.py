from .._utils import LAB_CIE_D65, _compute_masks
from ..typing import ChamberType
from typing import Optional, Tuple
import numpy as np


def compute_masks(bgr_img: np.ndarray, lab_img: Optional[np.ndarray], chamber_type: ChamberType) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    return _compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type, min_bright_threshould=255)


def compute_pmf(lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    #TODO Utilizar lab_to_normalized
    # Map a*b* coordinates of pixels to the CIE standard illuminant D65 and convert the result to indices.
    l_ind = ((lab_img[analyte_msk, 0] - (lab_white[0] - LAB_CIE_D65[0])) * 2.55).astype(np.int64)
    in_range = np.logical_and(0 <= l_ind, l_ind < 256)
    l_ind = l_ind[in_range]
    # Compute the frequency of each a*b* pair in the sample.
    pmf = np.zeros(256, np.float32)
    np.add.at(pmf, l_ind, 1)
    pmf /= pmf.sum()
    img_to_pmf = (np.stack(np.nonzero(analyte_msk), axis=1)[in_range, ...], l_ind)
    # Return the PMF of the projected L values in the image of the sample and extra data produced during the process.
    return pmf, img_to_pmf
