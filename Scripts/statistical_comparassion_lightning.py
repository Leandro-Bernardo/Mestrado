import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import chemical_analysis as ca
import shutil
import pandas as pd
import scipy
import math

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset, AlkalinityEstimationFunction
from chemical_analysis.chloride import ChlorideSampleDataset, ProcessedChlorideSampleDataset, ChlorideEstimationFunction
#from chemical_analysis.sulfate import SulfateSampleDataset, ProcessedSulfateSampleDataset, SulfateEstimationFunction
#from chemical_analysis.phosphate import PhosphateSampleDataset, ProcessedPhosphateSampleDataset, PhosphateEstimationFunction


if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
ANALYTE = "Alkalinity"
SKIP_BLANK = False
SKIP_SEPARATED_BLANK_FILES = False
USE_CHECKPOINT = True

if ANALYTE == "Alkalinity":
    EPOCHS = 3
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8

elif ANALYTE == "Chloride":
    EPOCHS = 3
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8

PMF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}Network.ckpt")
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", f"{ANALYTE}_Samples")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache_dir")

if SKIP_BLANK == True and SKIP_SEPARATED_BLANK_FILES == True:  #dont use blanks nor separated blanks
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank")
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "no_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "no_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "no_blank")
    SAVE_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "no_blank", "statistics")
    os.makedirs(SAVE_EXCEL_PATH, exist_ok=True)

elif SKIP_BLANK == False and SKIP_SEPARATED_BLANK_FILES == True: # use blanks, ignore separated blank files
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank")
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "with_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "with_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "with_blank")
    SAVE_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "with_blank", "statistics")
    os.makedirs(SAVE_EXCEL_PATH, exist_ok=True)

elif SKIP_BLANK == False and SKIP_SEPARATED_BLANK_FILES == False: # use separated blanks
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank")
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank")
    BLANK_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank")
    BLANK_IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "no_blank")
    BLANK_DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "processed_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "processed_blank")
    SAVE_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "processed_blank")
    SAVE_BLANK_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "processed_blank")
    os.makedirs(SAVE_EXCEL_PATH, exist_ok=True)
    os.makedirs(SAVE_BLANK_EXCEL_PATH, exist_ok=True)

else:
    raise Exception("Missmatch combinations \n SKIP_BLANK must be False for use separated blanks ")


LAST_CHECKPOINT = sorted(os.listdir(os.path.join(CHECKPOINT_ROOT, MODEL_VERSION)), key = lambda x: int(x.split('_')[-1]))[-1]
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, MODEL_VERSION, LAST_CHECKPOINT)

list_files = os.listdir(DESCRIPTORS_ROOT)
files_size = len(list_files)
train_split_size = int((files_size // 2) * DATASET_SPLIT)
test_split_size = int((files_size // 2) - train_split_size)

print('Using this checkpoint:', CHECKPOINT_PATH)
print('Using this official model:', PMF_MODEL_PATH)


# loads data and splits into training and testing for the descriptor based model

X_train_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size)], dim=0).to(device=device)
y_train_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size)], dim=0).to(device=device)

X_test_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size, train_split_size + test_split_size)], dim=0).to(device=device)
y_test_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size, train_split_size + test_split_size)], dim=0).to(device=device)

print(f"X_train_descriptors_model, y_train_descriptors_model size: {X_train_descriptors_model.size()}, {y_train_descriptors_model.size()}")
print(f"X_test_descriptors_model, y_test_descriptors_model size: {X_test_descriptors_model.size()}, {y_test_descriptors_model.size()}")

# makes batchers
train_loader = DataLoader(list(zip(X_train_descriptors_model, y_train_descriptors_model)), batch_size = BATCH_SIZE, shuffle= True)
eval_loader = DataLoader(list(zip(X_test_descriptors_model, y_test_descriptors_model)), batch_size = EVALUATION_BATCH_SIZE, shuffle = False)

# clears data from memory
del X_train_descriptors_model
del y_train_descriptors_model
del X_test_descriptors_model
del y_test_descriptors_model

torch.cuda.empty_cache()

if SKIP_SEPARATED_BLANK_FILES == False:
    blank_files = os.listdir(BLANK_DESCRIPTORS_ROOT)
    blank_files_size = len(blank_files)//2

    # loads blank files
    X_blank_descriptors_model = torch.cat([torch.load(os.path.join(BLANK_DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(int(blank_files_size/2))], dim=0).to(device=device)
    y_blank_descriptors_model = torch.cat([torch.load(os.path.join(BLANK_DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(int(blank_files_size/2) )], dim=0).to(device=device)

    print(f"X_blank_descriptors_model, y_blank_descriptors_model size: {X_blank_descriptors_model.size()}, {y_blank_descriptors_model.size()}")

    # makes batches
    blank_loader = DataLoader(list(zip(X_blank_descriptors_model, y_blank_descriptors_model)), batch_size = EVALUATION_BATCH_SIZE, shuffle = False)

    del X_blank_descriptors_model
    del y_blank_descriptors_model

    torch.cuda.empty_cache()


# PMF BASED MODEL
#preprocessing
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

#data preprocessing
samples = SampleDataset(
    base_dirs = SAMPLES_PATH,
    progress_bar = True,
    skip_blank_samples = SKIP_BLANK,
    skip_incomplete_samples = True,
    skip_inference_sample= True,
    skip_training_sample = False,
    verbose = True
)

if ANALYTE == "Alkalinity":
    processed_samples = ProcessedSampleDataset(
    dataset = samples,
    cache_dir = CACHE_PATH,
    num_augmented_samples = 0,
    progress_bar = True,
    transform = None, )

elif ANALYTE == "Chloride":
    processed_samples = ProcessedSampleDataset(
        dataset = samples,
        cache_dir = CACHE_PATH,
        num_augmented_samples = 0,
        progress_bar = True,
        transform = None,
        lab_mean= pca_stats[f"{ANALYTE}"]['lab_mean'],
        lab_sorted_eigenvectors = pca_stats[f"{ANALYTE}"]['lab_sorted_eigenvectors'])

#DESCRIPTOR BASED MODEL
# model definition
if ANALYTE == "Alkalinity":
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=448, out_features=256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=128),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=32),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=32, out_features=1)
                                ).to(device=device)

elif ANALYTE == "Chloride":
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=1472, out_features=1024),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=1024, out_features=512),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=512, out_features=256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=128),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=32),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=32, out_features=1)
                                ).to(device=device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# utilities functions and classes
def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze(1)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING_VALUE)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, loss_fn):

    model.eval()  # change model to evaluation mode

    partial_loss = []
    predicted_value = []
    expected_value = []
    correct_predictions = 0
    total_samples = len(eval_loader)

    with torch.no_grad():
        for X_batch, y_batch in eval_loader:

            y_pred = model(X_batch).squeeze(1)
            predicted_value.append(round(y_pred.item(), 2))

            expected_value.append(y_batch.item())

            loss = loss_fn(y_pred, y_batch)
            partial_loss.append(loss.item())

            #_, predicted = torch.max(y_pred, 1)
            #correct_predictions += (predicted == y_batch).sum().item()

   # accuracy = correct_predictions / total_samples

    return partial_loss, predicted_value, expected_value # ,accuracy

def get_sample_identity(sample, identity_path):
    information = []
    with open(os.path.join(identity_path, f"{sample}_identity.txt")) as f:
        for line in f:
            information.append(line)
    datetime, analyst_name, sample_prefix, blank_filename = information[0], information[1], information[2], information[3]
    return datetime.strip("\n").strip('"'), analyst_name.strip("\n").strip('"'), sample_prefix.strip("\n").strip('"'), blank_filename.strip("\n").strip('"')

class Statistics():

    def __init__(self, sample_predictions_vector, sample_expected_value):
        self.sample_predictions_vector = torch.tensor(sample_predictions_vector, dtype=torch.float32)
        self.sample_expected_value = torch.tensor(sample_expected_value, dtype=torch.float32)

        self.mean = torch.mean(self.sample_predictions_vector).item()
        self.median = torch.median(self.sample_predictions_vector).item()
        self.mode = scipy.stats.mode(np.array(sample_predictions_vector).flatten())[0]#torch.linalg.vector_norm(torch.flatten(self.sample_predictions_vector), ord = 5).item()
        self.variance = torch.var(self.sample_predictions_vector).item()
        self.std = torch.std(self.sample_predictions_vector).item()
        self.mad = scipy.stats.median_abs_deviation(np.array(sample_predictions_vector).flatten())
        self.min_value = torch.min(self.sample_predictions_vector).item()
        self.max_value = torch.max(self.sample_predictions_vector).item()
        self.absolute_error = torch.absolute(self.sample_predictions_vector - self.sample_expected_value)
        self.relative_error = (self.absolute_error/self.sample_expected_value)*100
        self.mae = torch.mean(self.absolute_error).item()
        self.mpe = torch.mean(self.relative_error).item()
        self.std_mae = torch.std(self.absolute_error).item()
        self.std_mpe = torch.std(self.relative_error).item()


def main():

    #DESCRIPTOR BASED MODEL
    #trains the descriptor based model
    print("Training time")
    for actual_epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)

        print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

    #evaluation of the descriptor based model
    print("Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model,eval_loader, loss_fn)
    predicted_value, expected_value = np.array(predicted_value), np.array(expected_value)

    sample_predicted_value = np.reshape(predicted_value, (test_split_size, -1))
    sample_expected_value = np.reshape(expected_value, (test_split_size, -1))

    if SKIP_SEPARATED_BLANK_FILES == False:
        print("Evaluating blank files")
        blank_partial_loss, blank_predicted_value, blank_expected_value = evaluate(model, blank_loader, loss_fn)
        blank_predicted_value, blank_expected_value = np.array(blank_predicted_value), np.array(blank_expected_value)

        sample_blank_predicted_value = np.reshape(blank_predicted_value, (blank_files_size, -1))
        sample_blank_expected_value = np.reshape(blank_expected_value, (blank_files_size, -1))

        blank_stats_dict = {}
        for i in range(0, sample_blank_predicted_value.shape[0]):
            stats = Statistics(blank_predicted_value[i], blank_expected_value[i])
            datetime, analyst_name, sample_prefix, blank_filename = get_sample_identity(f"sample_{i}", BLANK_IDENTITY_PATH)
            blank_stats_dict[sample_prefix] = {
                                               "expected value": np.unique(blank_expected_value[i])[0],
                                               "mean": stats.mean,
                                               "median": stats.median,
                                               "mode": stats.mode,
                                               "variance": stats.variance,
                                               "std": stats.std,
                                               "mad": stats.mad,
                                               "min": stats.min_value,
                                               "max": stats.max_value,
                                              }

    sample_stats_dict = {}
    for i in range(0, sample_predicted_value.shape[0] - 1):
        stats = Statistics(sample_predicted_value[i], sample_expected_value[i])
        datetime, analyst_name, sample_prefix, blank_filename = get_sample_identity(f"sample_{i+ train_split_size}", IDENTITY_PATH)
        sample_stats_dict[sample_prefix] = {
                                            "analyst_name": analyst_name,
                                            "datetime": datetime,
                                            "blank_id": blank_filename,
                                            "expected value": np.unique(sample_expected_value[i])[0],
                                            "estimated": 0,  #estimation from pmf based model
                                            "mean": stats.mean,
                                            "median": stats.median,
                                            "mode": stats.mode,
                                            "variance": stats.variance,
                                            "std": stats.std,
                                            "mad": stats.mad,
                                            "min": stats.min_value,
                                            "max": stats.max_value,
                                            #"mean absolute error": stats.mae,
                                            #"mean relative error": stats.mpe,
                                            #"std mean absolute error": stats.std_mae,
                                            #"std mean relative error": stats.std_mpe,
                                           }



    #creates a dataframe and then saves the xmls file
    df_stats = pd.DataFrame(sample_stats_dict).transpose()

    if SKIP_SEPARATED_BLANK_FILES == False:
        blank_df = pd.DataFrame(blank_stats_dict).transpose()
        for id in df_stats.index:
            blank_file_name = df_stats.loc[id, 'blank_id']
            df_stats.loc[id, "mean"] = df_stats.loc[id, "mean"] - blank_df.loc[blank_file_name, "mean"]
            df_stats.loc[id, "median"] = df_stats.loc[id, "median"] - blank_df.loc[blank_file_name, "median"]
            df_stats.loc[id, "variance"] = df_stats.loc[id, "variance"] + blank_df.loc[blank_file_name, "variance"]  #Var(X-Y) = Var(X) + Var(Y) - 2Cov(X,Y) ;  Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)
            df_stats.loc[id, "std"] = math.sqrt(df_stats.loc[id, "variance"])
        blank_df.to_excel(os.path.join(f"{SAVE_BLANK_EXCEL_PATH}", "blank_statistics.xlsx"))


    df_stats.to_excel(f"{SAVE_EXCEL_PATH}/statistics.xlsx" )

    #PMF BASED MODEL
    #evaluation of the pmf based model
    # estimation_func = AlkalinityEstimationFunction(checkpoint=os.path.join(os.path.dirname(__file__), "checkpoints", "AlkalinityNetwork.ckpt")).to("cuda")
    # estimation_func.eval()

    # pmf_model_prediction = {}
    # for i in range(train_split_size, train_split_size + test_split_size):  #takes only the test samples
    #     prediction = estimation_func(calibrated_pmf = torch.as_tensor(Y_test_pmf_model[i].calibrated_pmf, dtype = torch.float32, device = "cuda"))
    #     pmf_model_prediction[f"sample_{i}"] =  prediction

    print(" ")



if __name__ == "__main__":
    main()

