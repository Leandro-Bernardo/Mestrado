from ._model import Bisulfite2DEstimationFunction, Bisulfite2DIntervalNetwork, Bisulfite2DIntervalNetworkSqueezeNetStyle, Bisulfite2DNetwork, Bisulfite2DNetworkSqueezeNetStyle, Bisulfite2DNetworkVgg11Style, Bisulfite2DUpNetwork, Bisulfite2DNetworkUpVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import Bisulfite2DSampleDataset, ProcessedBisulfite2DSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "BisulfiteNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "BisulfitePcaStats.npz"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "BisulfiteUpNetwork.ckpt"))