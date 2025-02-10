from .._model import *
from ..typing import CalibratedDistributions, Distribution, Logits, Prediction
from collections import OrderedDict
from torch.nn import Module
from typing import Any, List, Tuple
import torch


class EmulsionNetwork(ContinuousNetwork):
    def __init__(self, backbone: Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = torch.nn.Sequential(OrderedDict([
            ("pre_processing", PreProcessing(**kwargs)),
            ("backbone", backbone),
            ("sigmoid", torch.nn.Sigmoid()),
            ("postprocessing", PostProcessing(**kwargs)),
        ]))

    def forward(self, calibrated_pmf: CalibratedDistributions) -> Prediction:
        return self.model(calibrated_pmf)

    @property
    def input_range(self) -> Tuple[float, float]:
        return tuple(map(float, self.model[0].input_range))  # type: ignore

    @property
    def input_roi(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(map(lambda bounds: tuple(map(int, bounds)), self.model[0].input_roi))  # type: ignore

    @property
    def training_mad(self) -> float:
        return float(self.model[-1].training_mad)  # type: ignore

    @property
    def training_median(self) -> float:
        return float(self.model[-1].training_median)  # type: ignore


class EmulsionNetworkMobileNetV3Style(EmulsionNetwork):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backbone=MobileNetV3Small(in_channels=1, num_classes=1), **kwargs)


class EmulsionNetworkSqueezeNetStyle(EmulsionNetwork):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backbone=SqueezeNet1_1(in_channels=1, num_classes=1), **kwargs)


class EmulsionNetworkVgg11Style(EmulsionNetwork):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backbone=Vgg11(in_channels=1, num_classes=1), **kwargs)


class EmulsionIntervalNetwork(IntervalNetwork):
    def __init__(self, backbone: Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = torch.nn.Sequential(OrderedDict([
            ("pre_processing", PreProcessing(**kwargs)),
            ("backbone", backbone),
        ]))

    def forward(self, calibrated_pmf: CalibratedDistributions) -> Logits:
        return self.model(calibrated_pmf)

    @property
    def input_range(self) -> Tuple[float, float]:
        return tuple(map(float, self.model[0].input_range))  # type: ignore

    @property
    def input_roi(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(map(lambda bounds: tuple(map(int, bounds)), self.model[0].input_roi))  # type: ignore

    @property
    def training_mad(self) -> float:
        return float(self.model[-1].training_mad)  # type: ignore

    @property
    def training_median(self) -> float:
        return float(self.model[-1].training_median)  # type: ignore


class EmulsionIntervalNetworkSqueezeNetStyle(EmulsionIntervalNetwork):
    def __init__(self, *, num_divisions: int, **kwargs: Any) -> None:
        super().__init__(backbone=SqueezeNet1_1(in_channels=1, num_classes=self.num_intervals(num_divisions)), num_divisions=num_divisions, **kwargs)


class PostProcessing(Module):
    def __init__(self, *, training_mad: float, training_median: float, **_) -> None:
        super().__init__()
        self.training_mad = torch.nn.Parameter(torch.as_tensor(training_mad, dtype=torch.float32), requires_grad=False)
        self.training_median = torch.nn.Parameter(torch.as_tensor(training_median, dtype=torch.float32), requires_grad=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(training_mad={self.training_mad.item()}, training_median={self.training_median.item()})"

    def forward(self, normalized_concentration: torch.Tensor) -> Prediction:
        normalized_concentration = normalized_concentration.squeeze(-1)
        concentration = (2.0 * normalized_concentration - 1.0) * (2.5758 * 1.4826 * self.training_mad) + self.training_median
        return concentration, normalized_concentration


class PreProcessing(Module):
    def __init__(self, *, input_range: Tuple[float, float], input_roi: Tuple[Tuple[int, int], Tuple[int, int]], **_) -> None:
        super().__init__()
        self.input_range = torch.nn.Parameter(torch.as_tensor(input_range, dtype=torch.float32), requires_grad=False)
        self.input_roi = torch.nn.Parameter(torch.as_tensor(input_roi, dtype=torch.int32), requires_grad=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_range={tuple(map(float, self.input_range))}, input_roi={tuple(map(lambda arg: tuple(map(int, arg)), self.input_roi))})"

    def forward(self, calibrated_pmf: CalibratedDistributions) -> torch.Tensor:
        normalized = (calibrated_pmf - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        roi = normalized[..., self.input_roi[0, 0]:self.input_roi[0, 1] + 1, self.input_roi[1, 0]:self.input_roi[1, 1] + 1]
        return roi.unsqueeze(1)


class EmulsionEstimationFunction(EstimationFunction):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(EmulsionNetwork, **kwargs)

    def _check_and_reshape_calibrated_pmf(self, **kwargs: CalibratedDistributions) -> Tuple[torch.Tensor, ...]:
        result: List[torch.Tensor] = list()
        for name, calibrated_pmf in kwargs.items():
            if calibrated_pmf is None or not isinstance(calibrated_pmf, torch.Tensor) or calibrated_pmf.dtype != torch.float32 or calibrated_pmf.dim() != 3:
                raise ValueError(f"The {name} argument is not a valid batch of calibrated distribution with shape (batch_size, size, size)")
            result.append(calibrated_pmf)
        return tuple(result)

    def _check_and_reshape_pmfs(self, **kwargs: Distribution) -> Tuple[torch.Tensor, ...]:
        result: List[torch.Tensor] = list()
        for name, pmf in kwargs.items():
            if pmf is None or not isinstance(pmf, torch.Tensor) or pmf.dtype != torch.float32 or pmf.dim() != 2:
                raise ValueError(f"The {name} argument is not a valid distribution with shape (size, size)")
            height, width = pmf.shape
            result.append(pmf.view(1, 1, height, width))
        return tuple(result)
