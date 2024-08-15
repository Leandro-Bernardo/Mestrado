import os
import torch
import torchvision
import json, yaml
import cv2

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vgg11, resnet50, vgg16
from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor

### Variables ###
# reads setting`s json
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    ANALYTE = settings["analyte"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    SKIP_BLANK = settings["skip_blank"]

feature_extractors_dict = {
                      "vgg11": vgg11(),
                      "resnet50": resnet50(),
                      }
if SKIP_BLANK:
    SKIP_BLANK = "no_blank"
else:
    SKIP_BLANK = "with_blank"

### main ###
def main():
    ### select layers for feature extraction ###
    # get the model object
    feature_extractor = feature_extractors_dict[FEATURE_EXTRACTOR]
    # print a summary of the model layers
    print(summary(feature_extractor.cuda(), (3, 224, 224)))
    # create the symbolic trace of torch.FX
    train_nodes, eval_nodes = get_graph_node_names(feature_extractor)
    # save the selected nodes
    nodes = []
    while True:
        node = input(f"\nSelected Nodes (leave blank to finish): ")
        if node == "":
            break
        nodes.append(node)

    print(f"List of nodes: \n{json.dumps(nodes)}")
    print(f"\nCNN blocks: {len(nodes)}")

    ### extract features ###
    # load an image
    load_path = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", f"{SKIP_BLANK}", "train")
    img = cv2.imread(os.path.join(load_path, "sample_1.png"))
    img = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_BGR2RGB)  # converts image to rgb format
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # reshape to (1, channels, H, W) (expected input shape)
    img = img.to("cuda") # change device
    # create the feature extractor
    feature_extraction = create_feature_extractor(
                                                model = feature_extractor,
                                                return_nodes= nodes
                                                )
    # extract features
    with torch.no_grad():  # disable grad calculation
        features = feature_extraction(img)

    # saves all the features in a dictionary
    features_dict = {f"feature{i}": feature[0] for i, (key,feature) in enumerate(features.items())}
    # get the dims form the first cnn layer
    dim, heigh, width = features_dict["feature0"].shape[0], features_dict["feature0"].shape[1], features_dict["feature0"].shape[2]
    # rescales to the size of first cnn
    reescaled_features = {feature_index: torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(feature_value).to(torch.float32) for feature_index, feature_value in features_dict.items() }
    sample_features = torch.cat([feature for feature in reescaled_features.values()], dim=0)
    rf = 20  # receptive field value for each side of image
    sample_features = sample_features[:, rf : sample_features.shape[1] - rf,  rf : sample_features.shape[2] - rf]
    sample_features = torch.permute(sample_features, (1, 2, 0))
    #sample_features = torch.reshape(sample_features, (-1, DESCRIPTOR_DEPTH))

    print(f"\nImage shape after first cnn block: \n{dim, heigh, width}")
    print(f"\nDescriptor Depth: \n{sample_features.shape[-1]}")

    # disabled auto save
    # settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["feature_list"] = nodes

    # with open(os.path.join('.', 'settings.json'), "w") as file:
    #     json.dump(settings, file, indent=6, separators=(",", ": "), sort_keys=False)


if __name__ == "__main__":
    main()
