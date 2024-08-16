import os
import cv2
import torch
import torchvision
import numpy as np
import json, yaml


from torch import FloatTensor, UntypedStorage
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vgg11, resnet50
from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor

### Variables ###
# reads setting`s json
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    ANALYTE = settings["analyte"]
    SKIP_BLANK = settings["skip_blank"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    CNN_BLOCKS = settings["cnn_blocks"]
    FEATURE_LIST = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["feature_list"]
    RECEPTIVE_FIELD_DIM = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["receptive_field_dim"]
    DESCRIPTOR_DEPTH = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["descriptor_depth"]
    CNN1_OUTPUT_SHAPE =  settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["cnn1_output_shape"]
    CNN1_OUTPUT_SIZE = CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE

if SKIP_BLANK:
    LOAD_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", "train")
    LOAD_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", "val")
    LOAD_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", "test")
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
else:
    LOAD_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", "train")
    LOAD_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", "val")
    LOAD_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", "test")
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")

feature_extractors_dict = {
                      "vgg11": vgg11(),
                      "resnet50": resnet50(),
                      }

TOTAL_TRAIN_SAMPLES = (int(len(os.listdir(LOAD_TRAIN_PATH))/3))
TOTAL_VAL_SAMPLES = (int(len(os.listdir(LOAD_VAL_PATH))/3))
TOTAL_TEST_SAMPLES = (int(len(os.listdir(LOAD_TEST_PATH))/3))

os.makedirs(SAVE_PATH, exist_ok = True)

### main ###
def main(load_path,  total_samples, stage):

    feature_extractor = feature_extractors_dict[FEATURE_EXTRACTOR]
    #loads images
    cropped_images = []

    for i in range(int(len(os.listdir(load_path))/3)):
            cropped_images.append(cv2.imread(os.path.join(load_path, f"sample_{i}.png")))

    #creates a symbolic trace from torch.FX
    train_nodes, eval_nodes = get_graph_node_names(feature_extractor)
    #print(train_nodes)

    #creates the feature extractor object
    feature_extraction = create_feature_extractor(
                                                model = feature_extractor,
                                                return_nodes= FEATURE_LIST
                                                )

    # extract the features


    # creat untyped storage object for save the descriptors
    nbytes_float32 = torch.finfo(torch.float32).bits//8
    descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(SAVE_PATH, f"descriptors_{stage}.bin"), shared = True, nbytes= (total_samples * CNN1_OUTPUT_SIZE * DESCRIPTOR_DEPTH) * nbytes_float32)).view(total_samples * CNN1_OUTPUT_SIZE, DESCRIPTOR_DEPTH)
    expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(SAVE_PATH, f"descriptors_anotation_{stage}.bin"), shared = True, nbytes= (total_samples * CNN1_OUTPUT_SIZE) * nbytes_float32)).view(total_samples * CNN1_OUTPUT_SIZE)

    # process the features from all images

    #for i, (sample_key, sample) in enumerate(image_tensor.items()):  #extrai caracteristicas das camadas de convolucao desejadas
    index = 0
    for i, _ in enumerate(cropped_images):
        print(f"extracting features from {stage} samples {i}/{len(cropped_images)}")
        img = cv2.cvtColor(cropped_images[i].astype('uint8'),cv2.COLOR_BGR2RGB) # select an image
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # reshape to (1, channels, H, W) (expected input shape)
        img = img.to("cuda")  # change device

        with torch.no_grad():
            features = feature_extraction(img)

        features_dict = {f"cnn_block{j+1}": feature.squeeze() for j, feature in enumerate(features.values())}

        heigh, width = features_dict["cnn_block1"].shape[1], features_dict["cnn_block1"].shape[2]

        #reescales the feature maps to the size of the first cnn block output
        reescaled_features = {feature_index: torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(feature_value).to(torch.float32) for feature_index, feature_value in features_dict.items() }

        #concatenates all imagens into a single tensor and makes a cropp based on receptive field
        sample_features = torch.cat([feature for feature in reescaled_features.values()], dim=0)
        rf = int((RECEPTIVE_FIELD_DIM - 1)/2)  # receptive field value for each side of image
        sample_features = sample_features[:, rf : sample_features.shape[1] - rf,  rf : sample_features.shape[2] - rf]
        sample_features = torch.permute(sample_features, (1, 2, 0))
        sample_features = torch.reshape(sample_features, (-1, DESCRIPTOR_DEPTH))
        #reades the analyte's concentration (groundtruth) and expands its (to match the descriptors' size)
        sample_theoretical_value = torch.tensor(float(open(os.path.join(load_path, f"sample_{i}.txt")).read())).expand(len(sample_features)).to(torch.float32)

        assert sample_features.shape[0] == CNN1_OUTPUT_SIZE
        assert sample_features.shape[1] == DESCRIPTOR_DEPTH
        #iterates over the descriptors and write its values into the Untyped Storage Object
        for j in range(sample_features.shape[0]):
            descriptors[index] = sample_features[j]
            expected_value[index] = sample_theoretical_value[j]
            index+=1
        print(f"Written lines: {index} \nExpected lines: {total_samples * CNN1_OUTPUT_SIZE}")
        assert index == total_samples * CNN1_OUTPUT_SIZE  "written lines is lesser than expected lines"

    with open(os.path.join(SAVE_PATH, f"metadata_{stage}.json"), "w") as file:
        json.dump({
            "total_samples": total_samples,
            "image_shape": CNN1_OUTPUT_SHAPE,
            "image_size": CNN1_OUTPUT_SIZE,
            "descriptor_depth": DESCRIPTOR_DEPTH
        }, file)

if __name__ == "__main__":

    feature_extractor = feature_extractors_dict[FEATURE_EXTRACTOR]
    print(summary(feature_extractor.cuda(), (3, 224, 224)))

    main(LOAD_TRAIN_PATH, TOTAL_TRAIN_SAMPLES, "train")
    main(LOAD_VAL_PATH, TOTAL_VAL_SAMPLES, "val")
    main(LOAD_TEST_PATH, TOTAL_TEST_SAMPLES, "test")