from ._model import PhEstimationFunction, PhNetwork, PhNetworkSqueezeNetStyle, PhNetworkVgg11Style, PhUpNetwork, PhNetworkUpVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import PhSampleDataset, ProcessedPhSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "PhNetwork.ckpt"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "PhUpNetwork.ckpt"))