import torch

import torch.nn as nn
from typing import Optional, Dict

class Model_1(torch.nn.Module):
    def __init__(self, descriptor_depth: int = 1472, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        device: str = "cuda"
        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features = descriptor_depth, out_features=1024),
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
    def __init__(self, descriptor_depth: int = 3904, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features= descriptor_depth, out_features=4096),
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
    def __init__(self, descriptor_depth: int = 1856, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.dropoutL1 = nn.Dropout(p = sweep_config["dropout_upper_layers"])  if sweep_config is not None else nn.Dropout(p=0.5)
        self.dropoutL2 = nn.Dropout(p = sweep_config["dropout_middle_layers"]) if sweep_config is not None else nn.Dropout(p=0.4)
        self.dropoutL3 = nn.Dropout(p = sweep_config["dropout_middle_layers"]) if sweep_config is not None else nn.Dropout(p=0.3)
        self.dropoutL4 = nn.Dropout(p = sweep_config["dropout_lower_layers"])  if sweep_config is not None else nn.Dropout(p=0.2)

        device: str = "cuda"
        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features = descriptor_depth, out_features=1024),
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
        x = self.dropoutL1(x)
        x = self.l2(x)
        x = self.dropoutL2(x)
        x = self.l3(x)
        x = self.dropoutL3(x)
        x = self.l4(x)
        x = self.dropoutL4(x)
        x = self.l5(x)
        x = self.output_layer(x)

        return x


class Model_4(torch.nn.Module):
    def __init__(self, descriptor_depth: int = 1856, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.dropoutL1 = nn.Dropout(p = sweep_config["dropout_upper_layers"])  if sweep_config is not None else nn.Dropout(p=0.5)
        self.dropoutL2 = nn.Dropout(p = sweep_config["dropout_middle_layers"]) if sweep_config is not None else nn.Dropout(p=0.4)
        self.dropoutL3 = nn.Dropout(p = sweep_config["dropout_lower_layers"])  if sweep_config is not None else nn.Dropout(p=0.2)

        device: str = "cuda"
        self.in_layer = nn.Sequential(
            torch.nn.Linear(in_features = 832, out_features=512),
           # nn.BatchNorm1d(1024),
            torch.nn.ReLU())
        self.l1 = nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=256),
           # nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.l2 = nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
           # nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.l3 = nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64),
          # nn.BatchNorm1d(64),
            torch.nn.ReLU())
        self.l4 = nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
           # nn.BatchNorm1d(32),
            torch.nn.ReLU())
        self.output_layer = nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=1))

    def forward(self, x: torch.Tensor):
        x = self.in_layer(x)
        x = self.l1(x)
        x = self.dropoutL1(x)
        x = self.l2(x)
        x = self.dropoutL2(x)
        x = self.l3(x)
        x = self.dropoutL3(x)
        x = self.l4(x)
        x = self.output_layer(x)

        return x


class Zero_Dawn(torch.nn.Module):
    def __init__(self, descriptor_depth: int, sweep_config: Dict, **kwargs) -> None:
        super().__init__()

        self.in_channels = descriptor_depth
        self.sweep_config = sweep_config

        self.dropoutUpperLayers = nn.Dropout(p=sweep_config["dropout_upper_layers"]) if sweep_config is not None else nn.Dropout(p=0.5)
        self.dropoutMiddleLayers = nn.Dropout(p=sweep_config["dropout_middle_layers"]) if sweep_config is not None else nn.Dropout(p=0.4)
        self.dropoutLowerLayers = nn.Dropout(p=sweep_config["dropout_lower_layers"]) if sweep_config is not None else nn.Dropout(p=0.2)

        self.num_hidden_layers = self.sweep_config["num_hidden_layers"]

        self.layers_neuron = {f"layer_{i}": self.sweep_config[f"layer_{i}"] for i in range(self.num_hidden_layers, 0, -1)} # neuron is the output param
        self.layers_neuron["output_layer"] = self.sweep_config["output_layer"]

        # assures first layer's neuron (output) is lower than the descriptor depth
        self.layers_neuron[f"input_layer"] = self.largest_power_of_2_less_than_or_equal_to(descriptor_depth)

        # assures lower layers have same or fewer neurons (output) than higher layers
        # if self.layers_neuron[f"layer_{self.num_hidden_layers}"] > self.layers_neuron[f"input_layer"] :
        #         self.layers_neuron[f"layer_{self.num_hidden_layers}"] = self.layers_neuron[f"input_layer"]
        for i in range(1, self.num_hidden_layers):
            if self.layers_neuron[f"layer_{i}"] > self.layers_neuron[f"layer_{i+1}"]:
                self.layers_neuron[f"layer_{i+1}"] = self.layers_neuron[f"layer_{i}"]

        # encapsulates inputs and outputs values in one object
        self.layersIO = {
                        f"layer_{i}": {
                            "in": self.layers_neuron[f"layer_{i+1}"] if i < self.num_hidden_layers else self.layers_neuron[f"input_layer"],
                            "out": self.layers_neuron[f"layer_{i}"],
                            "dropout": self.dropout(i)
                        } for i in range(self.num_hidden_layers, 0, -1)}
        self.layersIO["input_layer"] = {"in": descriptor_depth, "out": self.layers_neuron["input_layer"]}
        self.layersIO["output_layer"] = {"in":self.layersIO["layer_1"]["out"], "out": self.layers_neuron["output_layer"]}

        # builds the neural network ...
        self.sequential_layers = nn.ModuleDict()
        # ... input layer
        self.sequential_layers["input_layer"] = nn.Sequential(
                                                             torch.nn.Linear(in_features=self.layersIO["input_layer"]["in"], out_features=self.layersIO["input_layer"]["out"]),
                                                             torch.nn.ReLU(),
                                                             nn.Dropout(p=self.layersIO[f"layer_{self.num_hidden_layers}"]["dropout"]))
        # ... hidden layers
        for i in range(self.num_hidden_layers, 0, -1):
            self.sequential_layers[f"layer_{i}"] = nn.Sequential(
                                                                torch.nn.Linear(in_features=self.layersIO[f"layer_{i}"]["in"], out_features=self.layersIO[f"layer_{i}"]["out"]),
                                                                torch.nn.ReLU(),
                                                                nn.Dropout(p=self.layersIO[f"layer_{i}"]["dropout"]) if i > 1 else nn.Dropout(p=0))
        # ... output layer
        self.sequential_layers["output_layer"] = nn.Sequential(
                                                              torch.nn.Linear(in_features=self.layersIO["output_layer"]["in"], out_features=self.layersIO["output_layer"]["out"]))

    def forward(self, x: torch.Tensor):
        # input layer
        x = self.sequential_layers["input_layer"](x)
        # hidden layers
        for i in range(self.num_hidden_layers, 0, -1):
            x = self.sequential_layers[f"layer_{i}"](x)
        # output layer
        x = self.sequential_layers["output_layer"](x)
        return x

    #Return the largest power of 2 less than or equal to n (to be the input of the input_layer)
    def largest_power_of_2_less_than_or_equal_to(self, descriptor_depth):
        return 2**(descriptor_depth.bit_length() - 1)

    # dropout values for ...
    def dropout(self, layer_num):
        division = self.num_hidden_layers // 3
        # ... upper layers
        if layer_num >= 2 * division:
            return self.sweep_config["dropout_upper_layers"]
        # ... lower layers
        elif layer_num <= division:
            return self.sweep_config["dropout_lower_layers"]
        # ... middle layers
        else:
            return self.sweep_config["dropout_middle_layers"]

# class Best_Model_2blocks_resnet50_imgsize_448(torch.nn.Module):   #https://wandb.ai/uff-and-prograf/Chloride/runs/5b8r3eap
#     def __init__(self, descriptor_depth: int, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
#         super().__init__()

#         self.input_layer = torch.nn.Sequential(
#                                             torch.nn.Linear(in_features=320, out_features=256, bias=True),
#                                             torch.nn.ReLU(),
#                                             torch.nn.Dropout(p=0.413745579016662, inplace=False))
#         self.l7 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=256, out_features=128, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.413745579016662, inplace=False))
#         self.l6 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=128, out_features=128, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.413745579016662, inplace=False))
#         self.l5 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=128, out_features=64, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.2619022269924123, inplace=False))
#         self.l4 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=64, out_features=64, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.2619022269924123, inplace=False))
#         self.l3 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=64, out_features=16, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.10657501506295054, inplace=False))
#         self.l2 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=16, out_features=16, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.10657501506295054, inplace=False))
#         self.l1 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=16, out_features=16, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0, inplace=False))
#         self.output_layer =  torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=16, out_features=1, bias=True))

#     def forward(self, x: torch.Tensor):
#         x = self.input_layer(x)
#         x = self.l7(x)
#         x = self.l6(x)
#         x = self.l5(x)
#         x = self.l4(x)
#         x = self.l3(x)
#         x = self.l2(x)
#         x = self.l1(x)
#         x = self.output_layer(x)

#         return x

class Best_Model_2blocks_resnet50_imgsize_448(torch.nn.Module):   #https://wandb.ai/uff-and-prograf/Chloride/runs/ffck0k6v
    def __init__(self, descriptor_depth: int, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                            torch.nn.Linear(in_features=320, out_features=256, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.4248358907730576, inplace=False))
        self.l7 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=256, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4248358907730576, inplace=False))
        self.l6 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4248358907730576, inplace=False))
        self.l5 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=64, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4248358907730576, inplace=False))
        self.l4 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=64, out_features=64, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4248358907730576, inplace=False))
        self.l3 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=64, out_features=16, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.28188645604471424, inplace=False))
        self.l2 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=16, out_features=4, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.08957535555787653, inplace=False))
        self.l1 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=4, out_features=4, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0, inplace=False))
        self.output_layer =  torch.nn.Sequential(
                                      torch.nn.Linear(in_features=4, out_features=1, bias=True))

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.l7(x)
        x = self.l6(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)

        return x


class Best_Model_3blocks_resnet50(torch.nn.Module):   #https://wandb.ai/uff-and-prograf/Chloride/runs/yhq4t8ow
    def __init__(self, descriptor_depth: int, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                            torch.nn.Linear(in_features=832, out_features=512, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l11 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=512, out_features=4096, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l10 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l9 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l8 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l7 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l6 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=2048, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l5 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2177579021351188, inplace=False))
        self.l4 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2177579021351188, inplace=False))
        self.l3 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=64, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.14220723888179568, inplace=False))
        self.l2 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=64, out_features=4, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.14220723888179568, inplace=False))
        self.l1 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=4, out_features=4, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0, inplace=False))
        self.output_layer =  torch.nn.Sequential(
                                      torch.nn.Linear(in_features=4, out_features=1, bias=True))

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.l11(x)
        x = self.l10(x)
        x = self.l9(x)
        x = self.l8(x)
        x = self.l7(x)
        x = self.l6(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)

        return x


# class Best_Model_3blocks_resnet50_imgsize_448(torch.nn.Module):   #https://wandb.ai/uff-and-prograf/Chloride/runs/yprpd00k
#     def __init__(self, descriptor_depth: int, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
#         super().__init__()

#         self.input_layer = torch.nn.Sequential(
#                                             torch.nn.Linear(in_features=832, out_features=512, bias=True),
#                                             torch.nn.ReLU(),
#                                             torch.nn.Dropout(p=0.4324747397573356, inplace=False))
#         self.l10 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=512, out_features=512, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.4324747397573356, inplace=False))
#         self.l9 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=512, out_features=512, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.4324747397573356, inplace=False))
#         self.l8 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=512, out_features=512, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.4324747397573356, inplace=False))
#         self.l7 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=512, out_features=128, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.4324747397573356, inplace=False))
#         self.l6 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=128, out_features=128, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.4324747397573356, inplace=False))
#         self.l5 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=128, out_features=64, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.2275198340176682, inplace=False))
#         self.l4 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=64, out_features=32, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.2275198340176682, inplace=False))
#         self.l3 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=32, out_features=32, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.1326385125068351, inplace=False))
#         self.l2 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=32, out_features=32, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0.1326385125068351, inplace=False))
#         self.l1 = torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=32, out_features=32, bias=True),
#                                       torch.nn.ReLU(),
#                                       torch.nn.Dropout(p=0, inplace=False))
#         self.output_layer =  torch.nn.Sequential(
#                                       torch.nn.Linear(in_features=32, out_features=1, bias=True))

    # def forward(self, x: torch.Tensor):
    #     x = self.input_layer(x)
    #     x = self.l10(x)
    #     x = self.l9(x)
    #     x = self.l8(x)
    #     x = self.l7(x)
    #     x = self.l6(x)
    #     x = self.l5(x)
    #     x = self.l4(x)
    #     x = self.l3(x)
    #     x = self.l2(x)
    #     x = self.l1(x)
    #     x = self.output_layer(x)

    #     return x

class Best_Model_3blocks_resnet50_imgsize_448(torch.nn.Module):   #https://wandb.ai/uff-and-prograf/Chloride/runs/yprpd00k
    def __init__(self, descriptor_depth: int, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                            torch.nn.Linear(in_features=832, out_features=512, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l10 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=512, out_features=4096, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l9 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l8 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l7 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=2048, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l6 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l5 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=64, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2275198340176682, inplace=False))
        self.l4 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=64, out_features=32, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2275198340176682, inplace=False))
        self.l3 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=32, out_features=32, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.1326385125068351, inplace=False))
        self.l2 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=32, out_features=32, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.1326385125068351, inplace=False))
        self.l1 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=32, out_features=32, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0, inplace=False))
        self.output_layer =  torch.nn.Sequential(
                                      torch.nn.Linear(in_features=32, out_features=1, bias=True))

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.l10(x)
        x = self.l9(x)
        x = self.l8(x)
        x = self.l7(x)
        x = self.l6(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)

        return x
class Best_Model_4blocks_resnet50(torch.nn.Module):
    def __init__(self, descriptor_depth: int, sweep_config: Optional[Dict] = None, device: str = "cuda", **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                            torch.nn.Linear(in_features=1856, out_features=1024, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l10 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=1024, out_features=1024, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l9 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=1024, out_features=512, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l8 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=512, out_features=512, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l7 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=512, out_features=512, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l6 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=512, out_features=256, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.36159908287354814, inplace=False))
        self.l5 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=256, out_features=256, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2177579021351188, inplace=False))
        self.l4 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=256, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.2177579021351188, inplace=False))
        self.l3 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=128, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.14220723888179568, inplace=False))
        self.l2 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=128, out_features=32, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.14220723888179568, inplace=False))
        self.l1 = torch.nn.Sequential(
                                      torch.nn.Linear(in_features=32, out_features=8, bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0, inplace=False))
        self.output_layer =  torch.nn.Sequential(
                                      torch.nn.Linear(in_features=8, out_features=1, bias=True))

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.l10(x)
        x = self.l9(x)
        x = self.l8(x)
        x = self.l7(x)
        x = self.l6(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)

        return x


