import torch
import os
import numpy as np
#import matplotlib.pyplot as plt
#import cv2
import chemical_analysis as ca
#import shutil
import pandas as pd
import scipy
import math
import json, yaml

from tqdm import tqdm

from models import alkalinity, chloride

#from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch import FloatTensor, UntypedStorage
from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset, AlkalinityEstimationFunction
from chemical_analysis.chloride import ChlorideSampleDataset, ProcessedChlorideSampleDataset, ChlorideEstimationFunction
#from chemical_analysis.sulfate import SulfateSampleDataset, ProcessedSulfateSampleDataset, SulfateEstimationFunction
#from chemical_analysis.phosphate import PhosphateSampleDataset, ProcessedPhosphateSampleDataset, PhosphateEstimationFunction
from typing import Tuple, List, Dict

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"


### Variables for descriptor based model ###
# reads setting`s yaml
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    # global variables
    ANALYTE = settings["analyte"]
    SKIP_BLANK = settings["skip_blank"]
    PROCESS_BLANK_FILES_SEPARATEDLY = settings["process_blank_files_separatedly"]
    MODEL_VERSION = settings["chosen_model"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    CNN_BLOCKS = settings["cnn_blocks"]
    IMAGE_SIZE = settings["image_size"]

    # training hyperparams variables
    MAX_EPOCHS = settings["models"]["max_epochs"]
    LR = settings["models"]["learning_rate"]
    LOSS_FUNCTION = settings["models"]["loss_function"]
    GRADIENT_CLIPPING = 0.24487266642640568 #settings["models"]["gradient_clipping"]
    BATCH_SIZE = 12320#settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["cnn1_output_shape"]**2   # uses all the descriptors from an single image as a batch
    BATCH_NORM = settings["models"]["batch_normalization"]

    # evaluation variables
    #EPOCHS = 1  #training epochs. Disabled
    EVALUATION_BATCH_SIZE = 1
    IMAGES_TO_EVALUATE = settings["statistical_analysis"]["images_to_evaluate"]
    RECEPTIVE_FIELD_DIM = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["receptive_field_dim"]
    DESCRIPTOR_DEPTH = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["descriptor_depth"]

### Variables for pmf based model ###
# checkpoint path
PMF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}Network.ckpt")
# images path to process
SAMPLES_ROOT = os.path.join(os.path.dirname(__file__), "..", f"{IMAGES_TO_EVALUATE}_samples")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache_dir")

## Descriptor based model setup ##
networks_choices = {"Alkalinity":{"model_1": alkalinity.Model_1(),
                                  "model_2": alkalinity.Model_2()},
                    "Chloride": {"model_1": chloride.Model_1(),
                                 "model_2": chloride.Model_2(),
                                 "model_3": chloride.Model_3(),
                                 "best_model_4blocks_resnet50": chloride.Best_Model_4blocks_resnet50(DESCRIPTOR_DEPTH),
                                 "best_model_3blocks_resnet50": chloride.Best_Model_3blocks_resnet50(DESCRIPTOR_DEPTH),
                                 #"best_model_2blocks_resnet50": chloride.Best_Model_2blocks_resnet50(DESCRIPTOR_DEPTH),
                                 "best_model_2blocks_resnet50_img_size_448": chloride.Best_Model_2blocks_resnet50_imgsize_448(DESCRIPTOR_DEPTH),
                                 "best_model_2blocks_resnet50_img_size_448": chloride.Best_Model_3blocks_resnet50_imgsize_448(DESCRIPTOR_DEPTH)}}
MODEL_NETWORK = networks_choices[ANALYTE][MODEL_VERSION].to("cuda")

loss_function_choices = {"mean_squared_error": torch.nn.MSELoss()}
LOSS_FUNCTION = loss_function_choices[LOSS_FUNCTION]

## PMF based model setup ##
dataset_processor = {"Alkalinity":{"dataset": AlkalinitySampleDataset, "processed_dataset": ProcessedAlkalinitySampleDataset},
                     "Chloride": {"dataset": ChlorideSampleDataset, "processed_dataset": ProcessedChlorideSampleDataset},
                     #"Sulfate": {"dataset": SulfateSampleDataset, "processed_dataset": ProcessedSulfateSampleDataset},
                     #"Phosphate": {"dataset": PhosphateSampleDataset, "processed_dataset": ProcessedPhosphateSampleDataset},
                    }

pca_stats = {
             #"Alkalinity": {"lab_mean": np.load(ca.alkalinity.PCA_STATS)['lab_mean'], "lab_sorted_eigenvectors": np.load(ca.alkalinity.PCA_STATS)['lab_sorted_eigenvectors']},
             "Chloride"  : {"lab_mean": np.load(ca.chloride.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.chloride.PCA_STATS)['lab_sorted_eigenvectors']},
             #"Sulfate"   : {"lab_mean": np.load(ca.sulfate.PCA_STATS)['lab_mean']   , "lab_sorted_eigenvectors": np.load(ca.sulfate.PCA_STATS)['lab_sorted_eigenvectors']},
             #"Phosphate" : {"lab_mean": np.load(ca.phosphate.PCA_STATS)['lab_mean'] , "lab_sorted_eigenvectors": np.load(ca.phosphate.PCA_STATS)['lab_sorted_eigenvectors']}
            }

SampleDataset = dataset_processor[f"{ANALYTE}"]["dataset"]
ProcessedSampleDataset = dataset_processor[f"{ANALYTE}"]["processed_dataset"]

if SKIP_BLANK == True and  PROCESS_BLANK_FILES_SEPARATEDLY == False:  # dont use blanks
    # model path
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)(img_size_{IMAGE_SIZE})")
    # data paths
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", f"{IMAGES_TO_EVALUATE}")
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "no_blank", f"{IMAGES_TO_EVALUATE}")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}",  "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    # save path
    SAVE_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}" , "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)", MODEL_VERSION, f"{IMAGES_TO_EVALUATE}", "statistics")

elif SKIP_BLANK == False and PROCESS_BLANK_FILES_SEPARATEDLY == False:  # use blanks and process it together
    # model paths
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)(img_size_{IMAGE_SIZE})")
    # data paths
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", f"{IMAGES_TO_EVALUATE}")
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "with_blank", f"{IMAGES_TO_EVALUATE}")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}",  "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    # save path
    SAVE_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}" , "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)", MODEL_VERSION, f"{IMAGES_TO_EVALUATE}", "statistics")

elif SKIP_BLANK == False and PROCESS_BLANK_FILES_SEPARATEDLY == True:  # process blanks separatedly
    # model path
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)(img_size_{IMAGE_SIZE})")
    # data paths
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", f"{IMAGES_TO_EVALUATE}")
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "no_blank", f"{IMAGES_TO_EVALUATE}")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}",  "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    # blank files paths
    BLANK_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank", f"{IMAGES_TO_EVALUATE}")
    BLANK_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank", f"{IMAGES_TO_EVALUATE}")
    BLANK_DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "processed_blank", f"{IMAGES_TO_EVALUATE}")
    # save path
    SAVE_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "processed_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)", MODEL_VERSION, f"{IMAGES_TO_EVALUATE}", "statistics")

elif SKIP_BLANK == True and PROCESS_BLANK_FILES_SEPARATEDLY == True:  # missmatch combination
    raise Exception('''
    Missmatch combinations
        Case: SKIP_BLANK == True and PROCESS_BLANK_FILES_SEPARATEDLY == True)

    OPTIONS:
        SKIP_BLANK must be  True   and  PROCESS_BLANK_FILES_SEPARATEDLY  False  for not to process blanks, or
        SKIP_BLANK must be  False  and  PROCESS_BLANK_FILES_SEPARATEDLY  False  for use blanks and process it together, or
        SKIP_BLANK must be  False  and  PROCESS_BLANK_FILES_SEPARATEDLY  True   for process blanks separatedly
        ''')


CHECKPOINT_FILENAME = f"checkpoint.ckpt"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FILENAME)

print('Using this checkpoint:', CHECKPOINT_PATH)
print('Using this official model:', PMF_MODEL_PATH)


# creates directories
os.makedirs(SAVE_EXCEL_PATH, exist_ok=True)


### DESCRIPTOR BASED MODEL FUNCTIONS ###
# loads datasets for  evaluation
def load_dataset(dataset_for_inference: str, descriptor_root: str = DESCRIPTORS_ROOT):
        # loads metadata
        with open(os.path.join(descriptor_root, f'metadata_{dataset_for_inference}.json'), "r") as file:
            metadata = json.load(file)
        total_samples = metadata['total_samples']
        image_size = metadata['image_size']
        descriptor_depth = metadata['descriptor_depth']
        nbytes_float32 = torch.finfo(torch.float32).bits//8

        #NOTE:
        # at the moment, descriptors are saved in the format (num samples, image_size, descriptors_depth), but they are read in format (num samples * image_size,descriptors_depth).
        # expected_value is saved in format (num samples, image_size), and read in format (num samples * image_size)
        descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(descriptor_root, f"descriptors_{dataset_for_inference}.bin"), shared = False, nbytes= (total_samples * image_size * descriptor_depth) * nbytes_float32)).view(total_samples * image_size, descriptor_depth)
        expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(descriptor_root, f"descriptors_anotation_{dataset_for_inference}.bin"), shared = False, nbytes= (total_samples * image_size) * nbytes_float32)).view(total_samples * image_size)

        return TensorDataset(descriptors.to("cuda"), expected_value.to("cuda"))

# if needed, fixs the state dict keys and loads it
def load_state_dict(model: torch.nn.Module, checkpoint_state_dict: Dict ):
    checkpoint_state_dict = dict(checkpoint_state_dict.items())
    if "model.in_layer" in checkpoint_state_dict.keys():
        new_state_dict = {key.replace('model.', '') : value for key, value in checkpoint_state_dict.items()}
    elif "model.sequential_layers.input_layer.0.weight" in  checkpoint_state_dict.keys():
        new_state_dict = {key.replace('model.sequential_layers.', '') : value for key, value in checkpoint_state_dict.items()}
        #new_state_dict = {key.replace('.0.', '.') : value for key, value in new_state_dict.items()}
        new_state_dict = {key.replace('layer_', 'l') : value for key, value in new_state_dict.items()}


    return model.load_state_dict(new_state_dict, strict=True)

# evaluates the model
def evaluate(
            model: torch.nn,
            eval_loader: TensorDataset,
            loss_fn: torch.nn = LOSS_FUNCTION) -> Tuple[np.array, np.array, np.array]:

    model.eval()  # change model to evaluation mode

    partial_loss = []
    predicted_value = []
    expected_value = []
    #total_samples = len(eval_loader)

    #evaluation step
    with torch.no_grad():
        for X_batch, y_batch in eval_loader:

            if BATCH_NORM:
                y_pred = model(X_batch.unsqueeze(0))
            else:
                y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            predicted_value.append(round(y_pred.item(), 2))

            expected_value.append(y_batch.item())

            loss = loss_fn(y_pred, y_batch)
            partial_loss.append(loss.item())

    partial_loss = np.array(partial_loss)
    predicted_value = np.array(predicted_value)
    expected_value = np.array(expected_value)

    return partial_loss, predicted_value, expected_value # ,accuracy

# gets samples informations (datetime, analyst_name, sample_prefix, blank_filename)
def get_sample_identity(sample, identity_path):
    information = []
    with open(os.path.join(identity_path, f"{sample}_identity.txt")) as f:
        for line in f:
            information.append(line)
    datetime, analyst_name, sample_prefix, blank_filename = information[0], information[1], information[2], information[3]
    return datetime.strip("\n").strip('"'), analyst_name.strip("\n").strip('"'), sample_prefix.strip("\n").strip('"'), blank_filename.strip("\n").strip('"')

# calculates some statistics
class Statistics():

    def __init__(self, sample_predictions_vector: List, sample_expected_value_vector: List):
        self.sample_predictions_vector = torch.tensor(sample_predictions_vector, dtype=torch.float32)
        self.sample_expected_value_vector = torch.tensor(sample_expected_value_vector, dtype=torch.float32)

        self.mean = torch.mean(self.sample_predictions_vector).item()
        self.median = torch.median(self.sample_predictions_vector).item()
        self.mode =  self.mode = torch.mode(self.sample_predictions_vector.flatten())[0].item()#scipy.stats.mode(np.array(sample_predictions_vector).flatten())[0]#torch.linalg.vector_norm(torch.flatten(self.sample_predictions_vector), ord = 5).item()
        self.variance = torch.var(self.sample_predictions_vector).item()
        self.std = torch.std(self.sample_predictions_vector).item()
        self.mad = scipy.stats.median_abs_deviation(np.array(sample_predictions_vector).flatten())
        self.min_value = torch.min(self.sample_predictions_vector).item()
        self.max_value = torch.max(self.sample_predictions_vector).item()
        self.absolute_error = torch.absolute(self.sample_predictions_vector - self.sample_expected_value_vector)
        self.relative_error = (self.absolute_error/self.sample_expected_value_vector)*100
        self.mae = torch.mean(self.absolute_error).item()
        self.mpe = torch.mean(self.relative_error).item()
        self.std_mae = torch.std(self.absolute_error).item()
        self.std_mpe = torch.std(self.relative_error).item()
        self.relative_error_mean = (torch.absolute(self.mean - self.sample_expected_value_vector[0])/sample_expected_value_vector[0]).item()*100
        self.relative_error_median = (torch.absolute(self.median - self.sample_expected_value_vector[0])/sample_expected_value_vector[0]).item()*100
        self.relative_error_mode = (torch.absolute(self.mode - self.sample_expected_value_vector[0])/sample_expected_value_vector[0]).item()*100


# Loads model checkpoint
model = MODEL_NETWORK
#optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
checkpoint = torch.load(CHECKPOINT_PATH)
try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)  # NOTE: Some checkpoints state dicts might not have the expected keys, as seen in  https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
except:
        load_state_dict(model, checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main(dataset_for_inference: str):
    dataset = load_dataset(dataset_for_inference)
    len_total_samples = int(len(os.listdir(os.path.join(ORIGINAL_IMAGE_ROOT)))/3) #TODO alterar isso para abrir a partir do json de metadados

    ## DISABLED ##
    # #pmf based model preprocessing
    # samples = SampleDataset(
    #     base_dirs = SAMPLES_ROOT,
    #     progress_bar = True,
    #     skip_blank_samples = SKIP_BLANK,
    #     skip_incomplete_samples = True,
    #     skip_inference_sample= True,
    #     skip_training_sample = False,
    #     verbose = True
    # )

    # if ANALYTE == "Alkalinity":
    #     processed_samples = ProcessedSampleDataset(
    #     dataset = samples,
    #     cache_dir = CACHE_PATH,
    #     num_augmented_samples = 0,
    #     progress_bar = True,
    #     transform = None, )

    # elif ANALYTE == "Chloride":
    #     processed_samples = ProcessedSampleDataset(
    #         dataset = samples,
    #         cache_dir = CACHE_PATH,
    #         num_augmented_samples = 0,
    #         progress_bar = True,
    #         transform = None,
    #         lab_mean= pca_stats[f"{ANALYTE}"]['lab_mean'],
    #         lab_sorted_eigenvectors = pca_stats[f"{ANALYTE}"]['lab_sorted_eigenvectors'])

    print("Descriptor based model. Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model=model, eval_loader=dataset)
    predicted_value, expected_value = np.array(predicted_value), np.array(expected_value)

    sample_predicted_value = np.reshape(predicted_value, (len_total_samples, -1))
    sample_expected_value = np.reshape(expected_value, (len_total_samples, -1))

    # TODO fix this
    ## evaluates blank samples if they were separated from training samples (if not, does nothing)
    # if PROCESS_BLANK_FILES_SEPARATEDLY == True:
    #     print("Evaluating blank files")
    #     blank_partial_loss, blank_predicted_value, blank_expected_value = evaluate(model, blank_loader, loss_fn)
    #     blank_predicted_value, blank_expected_value = np.array(blank_predicted_value), np.array(blank_expected_value)

    #     sample_blank_predicted_value = np.reshape(blank_predicted_value, (blank_files_size, -1))
    #     sample_blank_expected_value = np.reshape(blank_expected_value, (blank_files_size, -1))

    #     blank_stats_dict = {}
    #     for i in range(0, sample_blank_predicted_value.shape[0]):
    #         stats = Statistics(blank_predicted_value[i], blank_expected_value[i])
    #         datetime, analyst_name, sample_prefix, blank_filename = get_sample_identity(f"sample_{i}", BLANK_IDENTITY_PATH)
    #         blank_stats_dict[sample_prefix] = {
    #                                            "expected value": np.unique(blank_expected_value[i])[0],
    #                                            "mean": stats.mean,
    #                                            "median": stats.median,
    #                                            "mode": stats.mode,
    #                                            "variance": stats.variance,
    #                                            "std": stats.std,
    #                                            "mad": stats.mad,
    #                                            "min": stats.min_value,
    #                                            "max": stats.max_value,
    #                                           }

    sample_stats_dict = {}
    for i in range(0, sample_predicted_value.shape[0] - 1):
        stats = Statistics(sample_predicted_value[i], sample_expected_value[i])
        datetime, analyst_name, sample_prefix, blank_filename = get_sample_identity(f"sample_{i}", IDENTITY_PATH)
        sample_stats_dict[sample_prefix] = {
                                            "analyst_name": analyst_name,
                                            "datetime": datetime,
                                            "blank_id": blank_filename,
                                            "expected value": np.unique(sample_expected_value[i])[0],
                                            "estimated": 0,  #estimation from pmf based model
                                            "mean": stats.mean,
                                            "median": stats.median,
                                            "mode": stats.mode,
                                            "min": stats.min_value,
                                            "max": stats.max_value,
                                            "variance": stats.variance,
                                            "std": stats.std,
                                            "mad": stats.mad,
                                            #"mean absolute error": stats.mae,
                                            #"mean relative error": stats.mpe,
                                            #"std mean absolute error": stats.std_mae,
                                            #"std mean relative error": stats.std_mpe,
                                            "relative error (mean)": stats.relative_error_mean,
                                            "relative error (median)": stats.relative_error_median,
                                            "relative error (mode)": stats.relative_error_mode,
                                           }



    #creates a dataframe and then saves the xmls file
    df_stats = pd.DataFrame(sample_stats_dict).transpose()

    # TODO fix this
    # # fixes the predicted values if blank samples were separated from training samples (if not, does nothing)
    # if PROCESS_BLANK_FILES_SEPARATEDLY == True:
    #     blank_df = pd.DataFrame(blank_stats_dict).transpose()
    #     for id in df_stats.index:
    #         blank_file_name = df_stats.loc[id, 'blank_id']
    #         df_stats.loc[id, "mean"] = df_stats.loc[id, "mean"] - blank_df.loc[blank_file_name, "mean"]
    #         df_stats.loc[id, "median"] = df_stats.loc[id, "median"] - blank_df.loc[blank_file_name, "median"]
    #         df_stats.loc[id, "variance"] = df_stats.loc[id, "variance"] + blank_df.loc[blank_file_name, "variance"]  #Var(X-Y) = Var(X) + Var(Y) - 2Cov(X,Y) ;  Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)
    #         df_stats.loc[id, "std"] = math.sqrt(df_stats.loc[id, "variance"])
    #     blank_df.to_excel(os.path.join(f"{SAVE_EXCEL_PATH}", "blank_statistics.xlsx"))

    excel_filename = os.path.join(f"{SAVE_EXCEL_PATH}", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)(image_size_{IMAGE_SIZE}).xlsx")
    df_stats.to_excel(excel_filename)

    # #PMF BASED MODEL
    # #evaluation of the pmf based model
    # # estimation_func = AlkalinityEstimationFunction(checkpoint=os.path.join(os.path.dirname(__file__), "checkpoints", "AlkalinityNetwork.ckpt")).to("cuda")
    # # estimation_func.eval()

    # # pmf_model_prediction = {}
    # # for i in range(train_split_size, train_split_size + test_split_size):  #takes only the test samples
    # #     prediction = estimation_func(calibrated_pmf = torch.as_tensor(Y_test_pmf_model[i].calibrated_pmf, dtype = torch.float32, device = "cuda"))
    # #     pmf_model_prediction[f"sample_{i}"] =  prediction

    # print(" ")



if __name__ == "__main__":
    main(IMAGES_TO_EVALUATE)

