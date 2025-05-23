from ._model import Iron3EstimationFunction, Iron3Network, Iron3NetworkSqueezeNetStyle, Iron3NetworkVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import Iron3SampleDataset, ProcessedIron3SampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "Iron3Network.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "Iron3PcaStats.npz"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "Iron3UpNetwork.ckpt"))