import torch

import torch.nn as nn
from typing import Optional, Dict


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
