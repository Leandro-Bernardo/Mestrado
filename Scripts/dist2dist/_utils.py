import torch
import math
import numpy as np
import os
import json

from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode
from typing import Tuple, Optional
from torch import Tensor, FloatTensor, UntypedStorage
from tqdm import tqdm
from _const import Dist2DistCacheType

def rotate_and_translate(*, input:torch.Tensor, translate:Tuple[int,int], angle:float, center:Tuple[int,int], squeeze:bool=True) ->torch.Tensor:
    """Rotates and Translates a tensor considering a center
    Parameters
    ----------
    input: Tensor
        A 2d-tensor been it [H,W] and H>=W.
    translate: Tuple[int,int]
        A (x,y) compensation to consider.
    angle: float
        Angle in degrees from +180 to -180.
    center: Tuple[int,int]
        A (x,y) positioning to consider as the center of the tensor.
    Returns
    ----------
    Tensor
        a tensor with the transformation applied.
    """
    A = affine(
        img=input if (input.dim()>2) else input[None,:],
        scale=(1.0),
        translate=tuple(map(lambda x: x, translate)),
        shear=(0.0),
        angle=angle,
        center=tuple(map(lambda x: x, center)),
        interpolation=InterpolationMode.BILINEAR
    )

    if squeeze:
        return A.squeeze(0)
    else:
        return A

def rotate_and_translate_in_batch(*, A:torch.Tensor, translate:torch.Tensor, angle:torch.Tensor, center:torch.Tensor) -> torch.Tensor:
    AT = list(map(lambda img, t, a, c:
        rotate_and_translate(input=img, translate=tuple(map(lambda _t: _t.item(),t)), angle=a.item(), center=tuple(map(lambda _c: _c.item(), c)), squeeze=False)
    , A[:], translate[:], angle, center[:]))

    return torch.stack(AT)

def compute_affine_parameters(*, input:np.ndarray, roied:bool=False) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Calculates from a tensor its eigenvalues and eigenvectors using the eigenvector associated to the biggest eigenvalue the parameter needs to make a affine transformation considering translation and rotation relative to the tensor unbised center.

    Parameters
    ----------
    input: Tensor
        A 2d-tensor been it [H,W] and H>=W.

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray, float, np.ndarray]
        a tuple containing compensation relative to the center, translation, rotation and a transformed tensor. Center and Translation representes (x,y).
    """
    A = torch.tensor(input, dtype=torch.float32)
    if roied:
        h_size, w_size = A.shape
        w_pad = h_size//2
        h_pad = (((w_pad*2)+w_size)-h_size)//2
        A = torch.nn.ZeroPad2d((w_pad,w_pad,h_pad,h_pad))(A)

    height, width = A.shape
    mu, sigma = unbised_weighted_sample_mean_and_convariance(A=A, height=height, width=width)
    eigenvalue, eigenvectors = calculate_eigens(sigma=sigma)
    if eigenvalue[0] < 0:
        eigenvectors = -eigenvectors[0]

    # Realizar translação e rotação referente a origem calculada
    sample_center = mu.round()[0].type(torch.int)
    sample_center_y, sample_center_x = sample_center[1].item(), sample_center[0].item()

    # Compensação necessário para realizar a translação da PMF para o centro do tensor
    offset_trans_y, offset_trans_x = calculate_translation_offset_compensation(height=height, width=width, center_y=sample_center_y,center_x=sample_center_x)

    # Compensação de angulo para rotação em função  do eixo horizontal.
    theta = calculate_vector_angulation(V=eigenvectors[0], degrees=True)
    theta = -theta
    if abs(theta)>180 or abs(theta)<0:
        assert("Theta out of range -180 and 180 degrees")

    translate = np.array([offset_trans_x, offset_trans_y])
    angle = (theta)
    center = np.array([sample_center_x, sample_center_y])

    AT = rotate_and_translate(input=A, translate=translate, angle=angle, center=center)

    return center, translate, angle, AT.numpy()

def unbised_weighted_sample_mean_and_convariance(*, A:Tensor, height:int, width:int) -> Tuple[Tensor,Tensor]:
    # https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
    # Weighted Average E[x]
    v, u = torch.meshgrid((torch.arange(height), torch.arange(width)))
    x = torch.stack((u, v), dim=-1).view(-1, 2)
    w = A.view(-1)

    # Levando em consideração a fórmula relatada em (https://stats.stackexchange.com/questions/61225correct-equation-for-weighted-unbiased-sample-covariance)
    mu = (w[:, None] * x).sum(dim=0, keepdim=True) / w.sum() # Média com pesos

    x_ = x - mu # Tornando o ponto médio a origem
    sigma = (w * x_.T) @ x_ # Unbised weighted sample covariance (não precisa dividir por w.sum() porque os autovalores não são importantes)
    return mu, sigma

def calculate_eigens(*, sigma:Tensor) -> Tuple[Tensor,Tensor]:
    #sigma precisa seguir as restrições da torch.eigen.eig
    eigenvalue, eigenvector = torch.linalg.eig(sigma)
    eigenvalue = eigenvalue.type(torch.float32)
    eigenvector = eigenvector.type(torch.float32)
    if eigenvalue[0] < eigenvalue[1]:
        eigenvalue = torch.flip(eigenvalue,dims=[0])
        eigenvector = torch.flip(eigenvector,dims=[1])

    return eigenvalue, eigenvector.T # Returning vector by row

def calculate_translation_offset_compensation(*, height:int, width:int, center_y:int, center_x:int) -> Tuple[int,int]:
    height_center = height//2
    off_heignt_center = abs(height_center - center_y)
    if center_y > height_center:
        off_heignt_center *= -1
        off_heignt_center -= 1

    width_center = width//2
    off_width_center = abs(width_center - center_x)
    if center_x > width_center:
        off_width_center *= -1
        off_width_center -= 1

    return off_heignt_center, off_width_center

def calculate_vector_angulation(*, V:Tensor, degrees:bool=False) -> float:
    radians = torch.atan2(input=V[1], other=V[0])

    if degrees: # Convert radians in degrees
        return radians.item() * (180/math.pi)
    else:
        return radians.item()

def calculate_interpolation_factor(*, a: Optional[float], b: Optional[float], x: float) -> float:
    """Calculates the interpolation factor between two values

    Parameters
    ----------
    a: float
        A float indicating the lower limiter.
    b: float
        A float indicating the upper limiter.
    x: float
        A float indicating the target values.

    Returns
    ----------
    float
        the percentage value that represents the distance of the target from its limiters.
    """
    if a == None or b == None:
        return -1.0
    return (x - a) / (b - a)

def get_featured_roi_size(*, regressor: torch.nn.Module, input_size: Tuple) -> Tuple[int, ...]:
    """Gets from the output size (channel, hight and width) of the last feature layer.
    Parameters
    ----------
    regressor: torch.nn.Module
        A loaded model from which features will be generated.
    input_size: Tuple
        A tuple containing the hight and width of the PMF used as input.

    Returns
    ----------
    Tuple[int, int, int]:
        a tuple with channel, hight and width a tensor
    """
    features = []
    features_layer = regressor.model.backbone.features
    handle = features_layer[len(features_layer)-1].register_forward_hook(
        lambda layer, _, output: features.append(output)
    )
    regressor(torch.zeros(size=input_size).unsqueeze(0))
    handle.remove()
    return tuple(features[0].shape)

def _load_artificial_pmfs_from_cache(*, stage:str, dataset_root_dir:str):
    cache_root_dir = os.path.join(dataset_root_dir, "dist2dist")
    with open(os.path.join(cache_root_dir, f'{stage}-artificial_pmfs.json'), "r") as fin:
        header = json.load(fin)
    data_size = header['num_samples']
    nbytes_float32 = torch.finfo(torch.float32).bits // 8
    pmfs_shape = header['pmfs_shape']
    # Creates afine parameters UntypedStorage
    pmfs = FloatTensor(UntypedStorage.from_file(filename=os.path.join(cache_root_dir, f"{stage}-artificial_pmfs.bin"), nbytes=(np.prod(pmfs_shape).item() * nbytes_float32 * data_size), shared=True)).view(data_size, *pmfs_shape)
    expected_values = FloatTensor(UntypedStorage.from_file(filename=os.path.join(cache_root_dir, f"{stage}-expected_values.bin"), nbytes=(nbytes_float32*data_size), shared=True))
    return pmfs, expected_values

def _check_if_artificial_pmfs_exists(*, stage:str, dataset_root_dir:str):
    cache_root_dir = os.path.join(dataset_root_dir, "dist2dist")
    exists = [os.path.isfile(os.path.join(cache_root_dir, filename)) for filename in [f"{stage}-artificial_pmfs.bin", f"{stage}-expected_values.bin"]]
    return not exists.__contains__(False)