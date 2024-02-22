from .typing import ChamberType, Sample
from typing import Any, Dict, Final, List, Optional, Tuple, Union
import cv2
import math
import numpy as np
import scipy, scipy.ndimage, scipy.signal


LAB_CIE_D65: Final[np.ndarray] = np.asarray([100.0, 0.0, 0.0], dtype=np.float32)
LAB_SPACE_VERTICES: Final[np.ndarray] = np.asarray([[+100, +128, +128], [+100, -128, +128], [+100, -128, -128], [+100, +128, -128], [0, +128, +128], [0, -128, +128], [0, -128, -128], [0, +128, -128]], dtype=np.float32)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    shape = bgr.shape
    if len(shape) != 3:
        bgr = bgr.reshape((1, -1, 3))
    lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_img[:, :, 0] *= 100.0 / 255.0
    lab_img[:, :, 1:] -= 128.0
    return lab_img.reshape(shape) if len(shape) != 3 else lab_img



def whitebalance(lab: np.ndarray, lab_white: np.ndarray) -> np.ndarray:
    return lab - (lab_white - LAB_CIE_D65)



def lab_to_normalized(lab: np.ndarray, *, lab_white: np.ndarray, lab_mean: Optional[np.ndarray] = None, lab_sorted_eigenvectors: Optional[np.ndarray] = None, out_channels: Tuple[int, ...]) -> np.ndarray:
    if lab_mean is not None and lab_sorted_eigenvectors is not None:
        shape = lab.shape
        # Perform whitebalance.
        lab = whitebalance(lab, lab_white).reshape(-1, 3)
        # Map samples to PCA space.
        lab_pca = np.matmul(lab - lab_mean, lab_sorted_eigenvectors)[:, out_channels]
        # Compute the limits of PCA space.
        lab_pca_vertices = np.matmul(LAB_SPACE_VERTICES - lab_mean, lab_sorted_eigenvectors)[:, out_channels]
        min = lab_pca_vertices.min(axis=0)
        max = lab_pca_vertices.max(axis=0)
        # Normalize channels to [0, 1] range.
        return ((lab_pca - min) / (max - min)).reshape((*shape[:-1], len(out_channels)))
    elif lab_mean is None and lab_sorted_eigenvectors is None:
        # Perform whitebalance.
        lab = whitebalance(lab, lab_white)[..., out_channels]
        # Set the limist of L*a*b* space.
        min = LAB_SPACE_VERTICES[:, out_channels].min(axis=0)
        max = LAB_SPACE_VERTICES[:, out_channels].max(axis=0)
        # Normalize channels to [0, 1] range.
        return ((lab - min) / (max - min))
    else:
        raise ValueError("Check PCA arguments")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    shape = rgb.shape
    if len(shape) != 3:
        rgb = rgb.reshape((1, -1, 3))
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_img[:, :, 0] *= 100.0 / 255.0
    lab_img[:, :, 1:] -= 128.0
    return lab_img.reshape(shape) if len(shape) != 3 else lab_img


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    shape = lab.shape
    if len(shape) != 3:
        lab = lab.reshape((1, -1, 3))
    tmp_img = np.copy(lab)
    tmp_img[:, :, 1:] += 128.0
    tmp_img[:, :, 0] /= 100.0 / 255.0
    bgr_img = cv2.cvtColor(tmp_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return bgr_img.reshape(shape) if len(shape) != 3 else bgr_img


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    shape = lab.shape
    if len(shape) != 3:
        lab = lab.reshape((1, -1, 3))
    tmp_img = np.copy(lab)
    tmp_img[:, :, 1:] += 128.0
    tmp_img[:, :, 0] /= 100.0 / 255.0
    rgb_img = cv2.cvtColor(tmp_img.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb_img.reshape(shape) if len(shape) != 3 else rgb_img


def check_mutually_exclusive_kwargs(**kwargs: Optional[Any]) -> None:
    count = sum([value is not None for value in kwargs.values()])
    if count == 0:
        raise ValueError(f'One of the following arguments must be provided: {", ".join(sorted(kwargs.keys()))}')
    elif count > 1:
        raise ValueError(f'The following arguments are mutually exclusive: {", ".join(sorted(kwargs.keys()))}')


def compute_calibrated_pmf(blank_pmf: np.ndarray, sample_pmf: np.ndarray) -> np.ndarray:
    # Compute C = A - B, where A is the random variable representing the sample and B is the random variable representing the blank sample
    return np.abs(scipy.signal.fftconvolve(sample_pmf, np.flip(blank_pmf), mode="same"))


def _compute_masks_for_cuvette(bgr_img: np.ndarray, lab_img: Optional[np.ndarray])-> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    # Define a function to help finding the grid lines
    def find_grid(msk: np.ndarray, axis: int, sigma: float) -> np.ndarray:
        values = scipy.ndimage.gaussian_filter1d(msk.sum(axis=axis), sigma=sigma)
        peaks, _ = scipy.signal.find_peaks(values)
        return np.sort(peaks)
    # Convert the input image from BGR to L*a*b* and set the correct range for the channels, i.e., 0 ≤ L* ≤ 100, −127 ≤ a* ≤ 127, −127 ≤ b* ≤ 127
    if lab_img is None:
        lab_img = bgr_to_lab(bgr_img)
    # Resize the input image
    height, width, _ = bgr_img.shape
    resized_height, resized_width = height // 4, width // 4
    resized_bgr_img = cv2.resize(bgr_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    resized_lab_img = cv2.resize(lab_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    # Find the bright pixels and set a mask for them (most pixels are from the grid, but some of pixels may be from the cuvette's cap and from highlights)
    bright_img = ((resized_bgr_img.astype(np.float32) / 255.0).prod(axis=2) * 255.0).astype(np.uint8)
    _, bright_bw = cv2.threshold(bright_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_msk = bright_bw.astype(np.bool_)  # bright_msk.shape = (height, width)
    # Estimate the X and Y coordinates of the end-points of the grid
    COLUMNS_COUNT = 6
    ROWS_COUNT = 8
    sigma = resized_height / (COLUMNS_COUNT * 20)
    grid_x_top = find_grid(bright_msk[: resized_height // 2, :], axis=0, sigma=sigma)
    grid_x_bottom = find_grid(bright_msk[resized_height // 2 :, :], axis=0, sigma=sigma)
    sigma = resized_height / (ROWS_COUNT * 20)
    grid_y_left = find_grid(bright_msk[:, : resized_width // 2], axis=1, sigma=sigma)
    grid_y_right = find_grid(bright_msk[:, resized_width // 2 :], axis=1, sigma=sigma)
    # Compute a mask for the pixels' grid (here we remove the cuvettes' pixels and grid pixels on the top and bottom of it)
    LEFT_COLUMN = 2  # Indexing from left is safer
    RIGHT_COLUMN = -3  # Indexing from right is safer
    grid_bw = np.copy(bright_bw)
    if len(grid_x_top) >= max(LEFT_COLUMN - 1, abs(RIGHT_COLUMN)) and len(grid_x_bottom) >= max(LEFT_COLUMN - 1, abs(RIGHT_COLUMN)):
        # Erase the cuvette's column
        cv2.drawContours(grid_bw, [np.asarray(((grid_x_top[LEFT_COLUMN], 0), (grid_x_top[RIGHT_COLUMN], 0), (grid_x_bottom[RIGHT_COLUMN], resized_height - 1), (grid_x_bottom[LEFT_COLUMN], resized_height - 1)), dtype=np.int32)], -1, 0, cv2.FILLED) 
    grid_msk = grid_bw.astype(np.bool_)
    # Compute the reference L*a*b value for white as the median L*a*b* values among the grid pixels
    lab_white = np.median(resized_lab_img[grid_msk, :], axis=0)  # lab_white.shape = (3,)
    # Find the analyte pixels and set a mask for them
    TOP_ROW = -5  # Indexing from bottom is safer
    BOTTOM_ROW = -2  # Indexing from bottom is safer
    not_bright_msk = np.logical_not(bright_msk)
    ab_dist = np.linalg.norm(resized_lab_img[not_bright_msk, 1:] - lab_white[1:], axis=1)
    analyte_bw = np.zeros((resized_height, resized_width, 1), dtype=np.uint8)
    _, analyte_bw[not_bright_msk, :] = cv2.threshold((ab_dist * (255.0 / ab_dist.max())).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    analyte_bw = analyte_bw.squeeze(axis=2)
    if len(grid_y_left) >= max(abs(TOP_ROW), abs(BOTTOM_ROW)) and len(grid_y_right) >= max(abs(TOP_ROW), abs(BOTTOM_ROW)):
        cv2.drawContours(analyte_bw, [np.asarray(((0, 0), (0, grid_y_left[TOP_ROW]), (resized_width - 1, grid_y_right[TOP_ROW]), (resized_width - 1, 0),), dtype=np.int32), np.asarray(((0, grid_y_left[BOTTOM_ROW]), (0, resized_height - 1), (resized_width - 1, resized_height - 1), (resized_width - 1, grid_y_right[BOTTOM_ROW])), dtype=np.int32)], -1, 0, cv2.FILLED)
    cnts, _ = cv2.findContours(analyte_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    analyte_bw = np.zeros((resized_height, resized_width), np.uint8)
    cv2.drawContours(analyte_bw, [cnt], -1, 255, cv2.FILLED)
    analyte_bw[bright_msk] = 0
    # Resize the masks to the orinal input size
    bright_msk = cv2.resize(bright_bw, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    grid_msk = cv2.resize(grid_bw, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    analyte_msk = cv2.resize(analyte_bw, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    # Return the masks (bright, grid, and analyte), the L*a*b* version of the image, and the reference L*a*b value for white
    return (bright_msk, grid_msk, analyte_msk, np.ones(analyte_msk.shape, dtype=np.float32)), lab_img, lab_white


def _compute_masks_for_pot(bgr_img: np.ndarray, lab_img: Optional[np.ndarray], min_bright_threshould: Optional[int]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    # Convert the input image from BGR to L*a*b* and set the correct range for the channels, i.e., 0 ≤ L* ≤ 100, −127 ≤ a* ≤ 127, −127 ≤ b* ≤ 127
    if lab_img is None:
        lab_img = bgr_to_lab(bgr_img)
    # Resize the input image
    height, width, _ = bgr_img.shape
    resized_height, resized_width = height // 4, width // 4
    resized_bgr_img = cv2.resize(bgr_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    resized_lab_img = cv2.resize(lab_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    # Compute the bright image and find the main circle
    bright_img = ((resized_bgr_img.astype(np.float32) / 255.0).prod(axis=2) * 255.0).astype(np.uint8)
    bright_img_ = cv2.morphologyEx(bright_img, cv2.MORPH_OPEN, (3, 3), iterations=15)
    detected_circles = cv2.HoughCircles(cv2.GaussianBlur(bright_img_, (3, 3), 0), cv2.HOUGH_GRADIENT, 1, 1, param1=50, param2=30, minRadius=min(resized_height, resized_width) // 6, maxRadius=min(resized_height, resized_width) // 3)
    if detected_circles is not None:
        detected_circles = np.around(detected_circles).astype(np.uint16)
        x, y, r = detected_circles[0, 0]
        # Compute the mask for the grid
        grid_bw = bright_img.copy()
        cv2.circle(grid_bw, (x, y), r, 0, -1)
        _, grid_bw = cv2.threshold(grid_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Compute the mask for the internal portion of the pot
        GRID_HOLE_RADIUS = 1.75  # In centimeters
        POT_EXTERNAL_RADIUS = 1.5  # In centimeters
        POT_INTERNAL_RADIUS = 1.1  # In centimeters
        pot_bw = np.zeros(shape=(resized_height, resized_width), dtype=np.uint8)
        cv2.circle(pot_bw, (x, y), int(r * (((POT_EXTERNAL_RADIUS + POT_INTERNAL_RADIUS) - GRID_HOLE_RADIUS) / GRID_HOLE_RADIUS)), 255, -1)
        pot_msk = pot_bw.astype(np.bool_)  # pot_msk.shape = (height, width)
        # Compute the mask for highlights
        bright_bw = np.zeros(shape=(resized_height, resized_width), dtype=np.uint8)
        analyte_thresh_options = cv2.THRESH_BINARY
        if min_bright_threshould is None:
            min_bright_threshould = 0
            analyte_thresh_options += cv2.THRESH_OTSU
        bright_bw[pot_msk] = cv2.threshold(bright_img[pot_msk], min_bright_threshould, 255, analyte_thresh_options,)[1][:, 0]
        # Compute the mask for the analyte
        analyte_bw = np.zeros(shape=(resized_height, resized_width), dtype=np.uint8)
        analyte_bw[np.logical_and(pot_msk, bright_bw != 255)] = 255
    else:
        bright_bw = np.zeros(shape=(resized_height, resized_width), dtype=np.uint8)
        grid_bw = np.zeros(shape=(resized_height, resized_width), dtype=np.uint8)
        analyte_bw = np.zeros(shape=(resized_height, resized_width), dtype=np.uint8)
    # Compute the reference L*a*b value for white as the median L*a*b* values among the grid pixels
    lab_white = np.median(resized_lab_img[grid_bw != 0, :], axis=0)  # lab_white.shape = (3,)
    # Resize the masks to the orinal input size
    bright_msk = cv2.resize(bright_bw, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    grid_msk = cv2.resize(grid_bw, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    analyte_msk = cv2.resize(analyte_bw, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    # Return the masks (bright, grid, analyte, and attention), the L*a*b* version of the image, and the reference L*a*b value for white
    return (bright_msk, grid_msk, analyte_msk), lab_img, lab_white


def _compute_masks(bgr_img: np.ndarray, lab_img: Optional[np.ndarray], chamber_type: ChamberType, min_bright_threshould: Optional[int]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    # Compute masks according to the chamber type
    if chamber_type is ChamberType.CUVETTE:
        return _compute_masks_for_cuvette(bgr_img, lab_img)
    elif chamber_type is ChamberType.POT:
        return _compute_masks_for_pot(bgr_img, lab_img, min_bright_threshould)
    else:
        raise ValueError(f"Invalid chamber type: {chamber_type}")
    

def correct_predicted_value(value: float, standard_volume: float, used_volume: float) -> float:
    return value * (standard_volume / used_volume)


def correct_theoretical_value(theoretical_value: float, standard_volume: float, used_volume: float, stock_factor: float) -> float:
    return theoretical_value * (used_volume / standard_volume) * stock_factor


def compute_theoretical_value(stock_value: float, aliquot: float, final_volume: float) -> float:
    return stock_value * (aliquot / final_volume) if aliquot != 0.0 else 0.0


def estimate_confidence_in_whitebalance(lab_img: np.ndarray, grid_msk: np.ndarray, whitebalance_stats: Union[str, Dict[str, Any]]) -> float:
    # Load whitebalance stats.
    if isinstance(whitebalance_stats, str):
        stats: Dict[str, Any] = np.load(whitebalance_stats)
    else:
        stats: Dict[str, Any] = whitebalance_stats
    mean: np.ndarray = stats["mean"]
    yaw: float = stats["yaw"]
    t_cov: np.ndarray = stats["x_aligned_cov"]
    inv_yaw = -yaw  # We need to rotate back covariance to align with the axis of the abscissas.
    cos_angle, sin_angle = np.cos(inv_yaw), np.sin(inv_yaw)
    # Extract grid pixels from the sample.
    lab_grid = lab_img[grid_msk, 1:]
    # The same with sample mean.
    sample_mean = np.mean(lab_grid, axis=0)
    t_sample_mean_x = -mean[0] * cos_angle + mean[1] * sin_angle + sample_mean[0] * cos_angle - sample_mean[1] * sin_angle
    # Reflect ellipsis, if sample is not on the negative side.
    if t_sample_mean_x > 0:
        t_sample_mean_x *= -1
    # The cummulative function through x-axis gives us the confidence.
    std_x = np.sqrt(2) * t_cov[0, 0]
    return 1 + math.erf(t_sample_mean_x / std_x)  # return 2 * cdf = [1 + erf((x - mu) / std * sqrt(2))]


def write_whitebalance_stats(samples: List[Sample], filepath: str) -> None:
    # Calculate some statistics, if not previously given.
    means, covs = list(), list()
    for sample in samples:
        bgr_img = cv2.imread(sample["fileName"], cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise RuntimeError(f'Can\'t load the file "{sample["fileName"]}"')
        (_, grid_msk, _, _), lab_img, _ = _compute_masks(bgr_img=bgr_img, lab_img=None, chamber_type=sample["chamberType"], min_bright_threshould=None)
        lab_grid = lab_img[grid_msk, 1:]
        means.append(np.mean(lab_grid, axis=0))
        covs.append(np.cov(lab_grid, rowvar=False))
    mean = np.mean(means, axis=0)
    cov = np.sum(covs, axis=0)  #TODO Pode isso?
    # Calculate first eigenvalue and eigenvector rotation angle.
    tau = (cov[1, 1] - cov[0, 0]) / (2 * cov[0, 1])
    t = np.sign(tau) / (np.abs(tau) + np.sqrt(1 + tau * tau))
    vector1_x = 1 / np.sqrt(1 + t * t)
    angle_rad = np.arctan2(-t * vector1_x, vector1_x)
    # Covariance after translation to origin, and axis alignment.
    cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
    t_matrix = np.asarray([
        [cos_angle, -sin_angle,  (mean[1] * sin_angle - mean[0] * cos_angle)],  # Let the Rotation and Translation matrices be, respectivelly, R and T.
        [sin_angle,  cos_angle, -(mean[0] * sin_angle + mean[1] * cos_angle)],  # C' = Transpose(R.T).C.(R.T)
        [0, 0, 1],
    ], dtype=np.float32)
    # Embed 'cov' matrix in homogeneous space and do the needed transformation.
    h_cov = np.array([[cov[0, 0], cov[0, 1], 0], [cov[0, 1], cov[1, 1], 0], [0, 0, 1]])
    t_cov = t_matrix.T.dot(h_cov).dot(t_matrix)
    np.savez_compressed(filepath, mean=mean, cov=cov, yaw=angle_rad, x_aligned_cov=t_cov)
