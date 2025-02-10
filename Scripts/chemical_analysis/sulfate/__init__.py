from ._model import SulfateEstimationFunction, SulfateNetwork, SulfateNetworkSqueezeNetStyle, SulfateNetworkVgg11Style, SulfateNetworkVgg19Style, SulfateUpNetwork, SulfateNetworkUpVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import SulfateSampleDataset, ProcessedSulfateSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "SulfateNetwork.ckpt"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "SulfateUpNetwork.ckpt"))