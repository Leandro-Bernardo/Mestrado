from typing import NamedTuple, Tuple, Type

from chemical_analysis import Network, UpNetwork
from chemical_analysis.sweep import ProcessedSampleDataset, SampleDataset

class AnalyteClasses(NamedTuple):
    analyte: str
    expected_range: Tuple[float, float]
    network_class: Type[Network]
    sample_dataset_class: Type[SampleDataset]
    processed_sample_dataset_class: Type[ProcessedSampleDataset]

class AnalyteUpClasses(NamedTuple):
    analyte: str
    network_class: Type[UpNetwork]
    sample_dataset_class: Type[SampleDataset]
    processed_sample_dataset_class: Type[ProcessedSampleDataset]

class AnalyteName(str):
    ALKALINITY = "alkalinity"
    CHLORIDE = "chloride"
    PHOSPHATE = "phosphate"
    SULFATE = "sulfate"
    IRON2 = "iron2"
    IRON3 = "iron32d"
    IRON_OXID = "iron_oxid"
    BISULFITE = "bisulfite2d"
    PH = "ph"

class ModelMode(str):
    TRAIN = 'Train'
    VALIDATION = 'Val'
    TEST = 'Test'

class WandbMode(str):
    ONLINE = 'online' # In this mode, the client sends data to the wandb server.
    OFFLINE = 'offline' # In this mode, instead of sending data to the wandb server, the client will store data on your local machine which can be later synced with the wandb sync command.
    DISABLE = 'disabled' # In this mode, the client returns mocked objects and prevents all network communication. The client will essentially act like a no-op. In other words, all logging is entirely disabled. However, stubs out of all the API methods are still callable. This is usually used in tests.

class DevMode(str):
    DEBUG = 'debug'
    PROD = 'production'
    GEN_PROD = 'generate_production'
    GEN_DEBUG = 'generate_debug'

class LightningStage(str):
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'

class Dist2DistCacheType(str):
    AUGMENTATION = 'augmentation'
    TRAINING = 'training'

class Dist2DistArtificialValues(str):
    EQUIDISTANT = 'equidistant'
    RANDOM = 'random'
    PERMUTED_EQUIDISTANT = 'permutation_equidistant'
    CYCLIC = 'cyclic'