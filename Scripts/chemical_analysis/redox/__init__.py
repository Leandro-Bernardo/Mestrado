from ._model import RedoxEstimationFunction, RedoxNetwork, RedoxNetworkMobileNetV3Style, RedoxNetworkSqueezeNetStyle, RedoxNetworkShuffleNetV2X10Style, RedoxNetworkShuffleNetV2X15Style, RedoxNetworkShuffleNetV2X20Style, RedoxNetworkVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import RedoxSampleDataset, ProcessedRedoxSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "RedoxNetwork.ckpt"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "RedoxUpNetwork.ckpt"))