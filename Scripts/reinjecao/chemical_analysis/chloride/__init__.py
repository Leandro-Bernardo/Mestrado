from ._model import ChlorideEstimationFunction, ChlorideIntervalNetwork, ChlorideIntervalNetworkSqueezeNetStyle, ChlorideNetwork, ChlorideNetworkSqueezeNetStyle, ChlorideNetworkVgg11Style, ChlorideNetworkUpVgg11Style, ChlorideUpNetwork
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import ChlorideSampleDataset, ProcessedChlorideSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "ChlorideNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "ChloridePcaStats.npz"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "ChlorideUpNetwork.ckpt"))