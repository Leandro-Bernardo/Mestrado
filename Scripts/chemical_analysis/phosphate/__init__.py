from ._model import PhosphateEstimationFunction, PhosphateNetwork, PhosphateNetworkSqueezeNetStyle, PhosphateNetworkVgg11Style, PhosphateUpNetwork, PhosphateNetworkUpVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import PhosphateSampleDataset, ProcessedPhosphateSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "PhosphateNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "PhosphatePcaStats.npz"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "PhosphateUpNetwork.ckpt"))