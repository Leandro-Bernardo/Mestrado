from ._model import EmulsionEstimationFunction, EmulsionIntervalNetwork, EmulsionIntervalNetworkSqueezeNetStyle, EmulsionNetwork, EmulsionNetworkSqueezeNetStyle, EmulsionNetworkVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import EmulsionSampleDataset, ProcessedEmulsionSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "EmulsionNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "EmulsionPcaStats.npz"))
