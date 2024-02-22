from ._model import AlkalinityEstimationFunction, AlkalinityNetwork, AlkalinityNetworkMobileNetV3Style, AlkalinityNetworkSqueezeNetStyle, AlkalinityNetworkVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "AlkalinityNetwork.ckpt"))
