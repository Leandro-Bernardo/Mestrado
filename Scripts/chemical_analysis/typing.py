from datetime import datetime
from enum import Enum, unique
from torch import Tensor
from typing import Any, List, Optional, Tuple, TypedDict


class App(TypedDict):
    packageName: str
    appName: str
    versionName: str


class SolutionComponent(TypedDict):
    name: str
    concentration: float
    concentrationUnit: str
    batch: str


class Solution(TypedDict):
    name: str
    components: List[SolutionComponent]


class AuxiliarySolutionComponent(SolutionComponent):
    function: str


class AuxiliarySolution(Solution):
    type: str


@unique
class ChamberType(Enum):
    CUVETTE = 0
    POT = 1


class Device(TypedDict):
    model: str
    manufacturer: str
    androidVersion: str


class StockAliquot(TypedDict):
    name: str
    finalVolume: float
    finalVolumeUnit: str
    aliquot: float
    aliquotUnit: str


class Stock(Solution):
    value: Optional[float]
    valueUnit: str
    aliquots: List[StockAliquot]


class Sample(TypedDict):
    # Properties filled by the SampleDataset class using data stored in the JSON file.
    app: App
    device: Device
    sourceStock: Stock
    sourceAliquot: StockAliquot
    stockFactor: float
    standardVolume: float
    usedVolume: float
    volumeUnit: str
    chamberType: ChamberType
    fileName: str
    extraFileNames: List[str]
    blankFileName: Optional[str]
    analystName: str
    notes: str
    datetime: datetime
    # Properties computed on the fly by the SampleDataset class.
    recordPath: Optional[str]
    name: str
    isBlankSample: bool
    isInferenceSample: bool
    isTrainingSample: bool
    theoreticalValue: Optional[float]
    correctedTheoreticalValue: Optional[float]
    blank: Optional["Sample"]
    # Properties computed on the fly by some subclass of SampleDataset.
    auxiliarySolutions: List[AuxiliarySolution]
    estimatedValue: Optional[float]
    valueUnit: str
    # Property set on the fly by data augmentation modules like ExpandedSampleDataset.
    referenceSample: Optional["Sample"]
    # Property set on the fly by any other peace of code.
    extra: Optional[Any]


CalibratedDistribution = Tensor   # Probability mass functions of one calibrated sample; dtype = torch.float32; shape = (height, width) or (size,).
CalibratedDistributions = Tensor  # Batch of probability mass functions of calibrated samples; dtype = torch.float32; shape = (batch_size, height, width) or (batch_size, size).
Distribution = Tensor             # Probability mass function of one sample; dtype = torch.float32; shape = (height, width) or (size,).
Distributions = Tensor            # Batch of probability mass functions of samples; dtype = torch.float32; shape = (batch_size, height, width) or (batch_size, size).
Intervals = Tensor                # Batch of predicted intervals; dtype = torch.float32; shape = (batch_size, 2).
Logits = Tensor                   # Bacth of class/interval predictions; dtype = torch.float32; shape = (batch_size, num_intervals)
Loss = Tensor                     # Batch of loss values; dtype = torch.float32; shape = (batch_size,).
NormalizedValues = Tensor         # Batch of normalized values in the [0, 1] range; dtype = torch.float32; shape = (batch_size,).
Value = Tensor                    # Predicted or expected value of one sample; dtype = torch.float32; shape = (,).
Values = Tensor                   # Batch of predicted or expected values; dtype = torch.float32; shape = (batch_size,).
Prediction = Tuple[Values, NormalizedValues]
