import torch
import torch.nn as nn
import torchvision.models as models

def calculate_receptive_field(model, layer_limit=None):
    def conv_rf(kernel_size, stride, padding, current_rf, current_stride):
        """
        Helper function to calculate receptive field for a single conv layer.
        """
        new_rf = current_rf + (kernel_size - 1) * current_stride
        new_stride = current_stride * stride
        return new_rf, new_stride

    # Load ResNet-50 model
    if model == 'resnet50':
        selected_model = models.resnet50(pretrained=False)
    elif model == 'vgg11':
        selected_model = models.vgg11(pretrained=False)
        raise ValueError(f"Model {model} is not supported")

    layers = list(selected_model.children())  # Get all layers
    if layer_limit:
        layers = layers[:layer_limit]  # Limit layers if needed

    # Initial receptive field and stride
    current_rf = 1
    current_stride = 1

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            # Update receptive field for convolutional layers
            current_rf, current_stride = conv_rf(layer.kernel_size[0], layer.stride[0], layer.padding[0], current_rf, current_stride)
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            # Update receptive field for pooling layers
            current_rf, current_stride = conv_rf(layer.kernel_size, layer.stride, layer.padding, current_rf, current_stride)
        elif isinstance(layer, nn.Sequential):
            # Traverse sequential blocks (used in bottleneck layers)
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Conv2d):
                    current_rf, current_stride = conv_rf(sub_layer.kernel_size[0], sub_layer.stride[0], sub_layer.padding[0], current_rf, current_stride)

    return current_rf

# Example usage:
# To calculate receptive field until layer 7 (which includes layers up to block 2 in ResNet-50):
rf = calculate_receptive_field('resnet50', layer_limit=12)
print(f"Receptive field: {rf}")
