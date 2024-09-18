import os
import cv2
import torch
import torchvision
import numpy as np
import json, yaml
import pandas as pd

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
    SAMPLES_PER_DATASET = settings["feature_extraction"]["samples_per_dataset"]

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

    #creates the feature extractor object
    feature_extraction = create_feature_extractor(
                                                model = feature_extractor,
                                                return_nodes= FEATURE_LIST
                                                )

    dataset_num = 0  #dataset identification
    treshold = 0     #treshold that avaliates samples in one dataset
    dataset_sample_idx = 0
    #expected_datasets = total_samples // SAMPLES_PER_DATASET
    #remainder = total_samples % SAMPLES_PER_DATASET

    # creates the mapper for samples
    samples_mapper = pd.DataFrame(0, columns=["descriptor_index", "dataset_descriptor_index", "sample_num", "dataset_num"], index=range(0,total_samples*CNN1_OUTPUT_SHAPE*CNN1_OUTPUT_SHAPE))

    # creates zeroes tensors to receive data
    dataset_unit_descriptors = torch.zeros((SAMPLES_PER_DATASET*CNN1_OUTPUT_SHAPE*CNN1_OUTPUT_SHAPE, DESCRIPTOR_DEPTH))
    dataset_unit_expected_values = torch.zeros((SAMPLES_PER_DATASET*CNN1_OUTPUT_SHAPE*CNN1_OUTPUT_SHAPE))
    # extract and process the features for all images
    for i, _ in enumerate(cropped_images):
        print(f"extracting features from {stage} samples {i+1}/{len(cropped_images)}")
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

        #writes the descriptors (and the anotations) values into the saving tensors (each tensor becomes a dataset, each dataset has SAMPLES_PER_DATASET samples)
        if i != len(cropped_images) - 1:
            starting_point_tensors = treshold * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE  #each sample i has (cnn1 output shape)^2 data
            ending_point_tensors = (treshold + 1) * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
        elif i == len(cropped_images) - 1:
            starting_point_tensors = treshold * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
            ending_point_tensors = starting_point_tensors + CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
        dataset_unit_descriptors[starting_point_tensors : ending_point_tensors] = sample_features
        dataset_unit_expected_values[starting_point_tensors : ending_point_tensors] = sample_theoretical_value

        #writes data for the mapper csv
        if i != len(cropped_images) - 1:
            starting_point_csv = i * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
            ending_point_csv = (i + 1) * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
            dataset_sample_idx_start = dataset_sample_idx * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
            dataset_sample_idx_end = (dataset_sample_idx + 1) * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
        elif i == len(cropped_images) - 1:
            starting_point_csv = i * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
            ending_point_csv = starting_point_csv + CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE - 1#- (CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE * remainder)
            dataset_sample_idx_start = dataset_sample_idx * CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE
            dataset_sample_idx_end = dataset_sample_idx_start + CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE - 1

        samples_mapper.loc[starting_point_csv : ending_point_csv, "descriptor_index"] = list(range(starting_point_csv, ending_point_csv + 1))
        samples_mapper.loc[starting_point_csv : ending_point_csv, "dataset_descriptor_index"] = list(range(dataset_sample_idx_start, dataset_sample_idx_end + 1))
        samples_mapper.loc[starting_point_csv : ending_point_csv, "sample_num"] = i
        samples_mapper.loc[starting_point_csv : ending_point_csv, "dataset_num"] = dataset_num

        treshold+=1
        dataset_sample_idx +=1

        if treshold >= SAMPLES_PER_DATASET or i == len(cropped_images) - 1:
            #saves current descriptors
            torch.save(dataset_unit_descriptors, os.path.join(SAVE_PATH, f"descriptors_{stage}_dataset_{dataset_num}.pt"))
            torch.save(dataset_unit_expected_values, os.path.join(SAVE_PATH, f"descriptors_anotation_{stage}_dataset_{dataset_num}.pt"))
            #reset tensors
            dataset_unit_descriptors = torch.zeros((SAMPLES_PER_DATASET*CNN1_OUTPUT_SHAPE*CNN1_OUTPUT_SHAPE, DESCRIPTOR_DEPTH))
            dataset_unit_expected_values = torch.zeros((SAMPLES_PER_DATASET*CNN1_OUTPUT_SHAPE*CNN1_OUTPUT_SHAPE))
            #reset treshold and changes dataset index
            dataset_num += 1
            treshold = 0
            dataset_sample_idx = 0


    with open(os.path.join(SAVE_PATH, f"metadata_{stage}.json"), "w") as file:
        json.dump({
            "total_samples": total_samples,
            "image_shape": CNN1_OUTPUT_SHAPE,
            "image_size": CNN1_OUTPUT_SIZE,
            "descriptor_depth": DESCRIPTOR_DEPTH
        }, file)

    samples_mapper.to_csv(os.path.join(SAVE_PATH, f"{stage}_mapping.csv"))

if __name__ == "__main__":

    feature_extractor = feature_extractors_dict[FEATURE_EXTRACTOR]
    print(summary(feature_extractor.cuda(), (3, 224, 224)))

    main(LOAD_TRAIN_PATH, TOTAL_TRAIN_SAMPLES, "train")
    main(LOAD_VAL_PATH, TOTAL_VAL_SAMPLES, "val")
    main(LOAD_TEST_PATH, TOTAL_TEST_SAMPLES, "test")