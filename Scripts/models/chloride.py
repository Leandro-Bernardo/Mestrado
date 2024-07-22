import torch

import torch.nn as nn

class Model_1(torch.nn.Module):
    def __init__(self, in_channels: int = 1472, device: str = "cuda"):
        super().__init__()

        device: str = "cuda"
        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features = in_channels, out_features=1024),
           # nn.BatchNorm1d(1024),
            torch.nn.ReLU())
        self.l1 = nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512),
           # nn.BatchNorm1d(512),
            torch.nn.ReLU())
        self.l2 = nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=256),
           # nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.l3 = nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
           # nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.l4 = nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64),
          # nn.BatchNorm1d(64),
            torch.nn.ReLU())
        self.l5 = nn.Sequential(
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
        x = self.l4(x)
        x = self.l5(x)
        x = self.output_layer(x)

        return x


class Model_2(torch.nn.Module):
    def __init__(self, in_channels: int  = 3904, device: str = "cuda"):
        super().__init__()

        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features= in_channels, out_features=4096),
            nn.BatchNorm1d(4096),
            torch.nn.ReLU())
        self.l1 = nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=2048),
            nn.BatchNorm1d(2048),
            torch.nn.ReLU())
        self.l2 = nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            torch.nn.ReLU())
        self.l3 = nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            torch.nn.ReLU())
        self.l4 = nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.l5 = nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.l6 = nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            torch.nn.ReLU())
        self.l7 = nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            torch.nn.ReLU())
        self.output_layer = nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=1))

    def forward(self, x: torch.Tensor):
        x = self.in_layer(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.output_layer(x)

        return x


class Model_3(torch.nn.Module):
    def __init__(self, in_channels: int = 1856, device: str = "cuda", dropout_p:float = 0.5):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_p)

        device: str = "cuda"
        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features = in_channels, out_features=1024),
           # nn.BatchNorm1d(1024),
            torch.nn.ReLU())
        self.l1 = nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512),
           # nn.BatchNorm1d(512),
            torch.nn.ReLU())
        self.l2 = nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=256),
           # nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.l3 = nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
           # nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.l4 = nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64),
          # nn.BatchNorm1d(64),
            torch.nn.ReLU())
        self.l5 = nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
           # nn.BatchNorm1d(32),
            torch.nn.ReLU())
        self.output_layer = nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=1))

    def forward(self, x: torch.Tensor):
        x = self.in_layer(x)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.dropout(x)
        x = self.l5(x)
        x = self.output_layer(x)

        return x