from ._model import SuspendedEstimationFunction, SuspendedIntervalNetwork, SuspendedIntervalNetworkSqueezeNetStyle, SuspendedNetwork, SuspendedNetworkSqueezeNetStyle, SuspendedNetworkVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import SuspendedSampleDataset, ProcessedSuspendedSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "SuspendedNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "SuspendedPcaStats.npz"))
