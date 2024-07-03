import torch.nn as nn
import torch

class Model_1(torch.nn.Module):
    def __init__():
        pass
    pass

class Model_2(torch.nn.Module):
    def __init__(self, in_channels: int = 448, device: str = "cuda"):
        super().__init__()

        device: str = "cuda"
        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features = in_channels, out_features=256),
           # nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.l1 = nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
           # nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.l2 = nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64),
           # nn.BatchNorm1d(64),
            torch.nn.ReLU())
        self.l3 = nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
           # nn.BatchNorm1d(32),
            torch.nn.ReLU())
        self.output_layer = nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=1))

    def forward(self, x: torch.Tensor):
        x = self.in_layer(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.output_layer(x)

        return x