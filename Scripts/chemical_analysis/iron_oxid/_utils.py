from .._utils import LAB_CIE_D65, LAB_SPACE_VERTICES, _compute_masks
from ..typing import ChamberType
from typing import Optional, Tuple
import numpy as np


def compute_masks(bgr_img: np.ndarray, lab_img: Optional[np.ndarray], chamber_type: ChamberType) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    return _compute_masks(bgr_img=bgr_img, lab_img=lab_img, chamber_type=chamber_type, min_bright_threshould=None)


def compute_pmf(lab_img: np.ndarray, analyte_msk: np.ndarray, lab_white: np.ndarray, lab_mean: np.ndarray, lab_sorted_eigenvectors: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    #TODO Utilizar lab_to_normalized
    # Map a*b* coordinates of pixels to the CIE standard illuminant D65 and convert the result to indices.
    lab_pca = lab_img[analyte_msk] - (lab_white - LAB_CIE_D65)
    # Prepare L*a*b* data to be reducted to single dimension.
    lab_matrix = (LAB_SPACE_VERTICES - lab_mean).dot(lab_sorted_eigenvectors)[:, 0]  #TODO Verificar se quando o PCA foi calculado, as imagens ja haviam passado por whitebalance
    lab_min = lab_matrix.min(axis=0)
    lab_max = lab_matrix.max(axis=0)
    lab_pca = (lab_pca - lab_mean).dot(lab_sorted_eigenvectors)[:, 0]
    lab_pca = (((lab_pca - lab_min) / (lab_max - lab_min)) * 255.0).astype(np.int64)
    in_range = np.logical_and(0 <= lab_pca, lab_pca < 256)
    lab_pca = lab_pca[in_range]
    # Compute the frequency of each a*b* pair in the sample.
    pmf = np.zeros(256, np.float32)
    np.add.at(pmf, lab_pca, 1)
    pmf /= pmf.sum()
    img_to_pmf = (np.stack(np.nonzero(analyte_msk), axis=1)[in_range, ...], lab_pca)  #TODO Testar in_range
    # Return the PMF of the projected L*a*b* entries in the image of the sample and extra data produced during the process.
    return pmf, img_to_pmf
