from ._model import IronOxidEstimationFunction,  IronOxidNetwork, IronOxidNetworkSqueezeNetStyle, IronOxidNetworkVgg11Style #IronOxidIntervalNetwork, IronOxidIntervalNetworkSqueezeNetStyle,
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import IronOxidSampleDataset, ProcessedIronOxidSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "IronOxidNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "IronOxidPcaStats.npz"))
