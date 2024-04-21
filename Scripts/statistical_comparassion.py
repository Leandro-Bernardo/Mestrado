import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import chemical_analysis
import shutil
import pandas as pd

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset, AlkalinityEstimationFunction

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
EPOCHS = 1
LR = 0.0001
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 1
GRADIENT_CLIPPING_VALUE = 0.5
MODEL_VERSION = 'model_1' #if len(os.listdir("./models")) == 0 else f'model_{len(os.listdir("./models"))}'
DATASET_SPLIT = 0.8

CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints")
DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors")
EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation")
LAST_CHECKPOINT = sorted(os.listdir(os.path.join(CHECKPOINT_ROOT, MODEL_VERSION)), key = lambda x: int(x.split('_')[-1]))[-1]
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, MODEL_VERSION, LAST_CHECKPOINT)

ANALYTE = 'Alkalinity'
PMF_MODEL_PATH = os.path.join(os.path.dirname(__file__), CHECKPOINT_ROOT, f"{ANALYTE}Network.ckpt")
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", f"{ANALYTE}_Samples" )
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache_dir")

print('Using this checkpoint:', CHECKPOINT_PATH)
print('Using this official model:', PMF_MODEL_PATH)


# loads data and splits into training and testing for the descriptor based model
list_files = os.listdir(DESCRIPTORS_ROOT)
files_size = len(list_files)
train_split_size = int((files_size // 2) * DATASET_SPLIT)
test_split_size = int((files_size // 2) - train_split_size)

X_train_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size)], dim=0).to(device=device)
y_train_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size)], dim=0).to(device=device)

X_test_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size, train_split_size + test_split_size)], dim=0).to(device=device)
y_test_descriptors_model = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size, train_split_size + test_split_size)], dim=0).to(device=device)

print(f"X_train_descriptors_model, y_train_descriptors_model size: {X_train_descriptors_model.size()}, {y_train_descriptors_model.size()}")
print(f"X_test_descriptors_model, y_test_descriptors_model size: {X_test_descriptors_model.size()}, {y_test_descriptors_model.size()}")

# makes batchers for training
train_loader = DataLoader(list(zip(X_train_descriptors_model, y_train_descriptors_model)), batch_size = BATCH_SIZE, shuffle= True)
eval_loader = DataLoader(list(zip(X_test_descriptors_model, y_test_descriptors_model)), batch_size = EVALUATION_BATCH_SIZE, shuffle = False)

# saves values for graph scale
min_value, max_value = float(torch.min(y_test_descriptors_model)), float(torch.max(y_test_descriptors_model))

# clears data from memory
del X_train_descriptors_model
del y_train_descriptors_model
del X_test_descriptors_model
del y_test_descriptors_model

torch.cuda.empty_cache()


# loads all data for the image pmf based model
# Y_test_pmf_model = AlkalinitySampleDataset(
#                                             base_dirs= SAMPLES_PATH,  #loads all data for processing
#                                             progress_bar = True,
#                                             skip_blank_samples = True,
#                                             skip_incomplete_samples = True,
#                                             skip_inference_sample= True,
#                                             skip_training_sample = False,
#                                             verbose = True
#                                             )

# Y_test_pmf_model = ProcessedAlkalinitySampleDataset(
#                                                     dataset = Y_test_pmf_model,
#                                                     cache_dir = CACHE_PATH,
#                                                     num_augmented_samples = 0,
#                                                     progress_bar = True,
#                                                     transform = None,
#                                                     )

# descriptor model based definition
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


class Statistics():

    def __init__(self, sample_predictions_vector):
        self.sample_predictions_vector = torch.tensor(sample_predictions_vector)
        self.mean = torch.mean(self.sample_predictions_vector).item()
        self.median = torch.median(self.sample_predictions_vector).item()
        self.mode = torch.mode(torch.flatten(self.sample_predictions_vector))[0].item()
        self.variance = torch.var(self.sample_predictions_vector).item()
        self.std = torch.std(self.sample_predictions_vector).item()
        self.min_value = torch.min(self.sample_predictions_vector).item()
        self.max_value = torch.max(self.sample_predictions_vector).item()

def main():

    #DESCRIPTOR BASED MODEL
    #training the descriptor based model
    print("Training time")
    for actual_epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)

        print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

    #evaluation of the descriptor based model
    print("Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model,eval_loader, loss_fn)
    predicted_value, expected_value = np.array(predicted_value), np.array(expected_value)

    absolute_error = np.absolute(predicted_value - expected_value)
    relative_error = (absolute_error/expected_value)*100

    absolute_error_from_samples = np.reshape(absolute_error, (test_split_size, -1))
    relative_error_from_samples = np.reshape(relative_error, (test_split_size, -1))

    #saves in dict format to create dataframes
    mean_absolute_error_from_samples = {}
    for i in range(train_split_size, train_split_size + test_split_size):
        mean_absolute_error_from_samples[f"sample_{i}"] = np.mean(absolute_error_from_samples[i])

    mean_percentage_error_from_samples = {}
    for i in rrange(train_split_size, train_split_size + test_split_size):
        mean_percentage_error_from_samples[f"sample_{i}"] = np.mean(relative_error_from_samples[i])


    #mean_absolute_error_from_samples = {f"sample_{i}": mean(value) for i, values in enumerate(absolute_error_from_samples)}
    #mean_percentage_error_from_samples =  {f"sample_{i}": mean(value) for i, values in enumerate(relative_error_from_samples)}
    print(mean_absolute_error_from_samples, mean_percentage_error_from_samples)

    #creates a dataframe and then saves the xmls file
    dataframe_mae = pd.DataFrame(mean_absolute_error_from_samples)
    dataframe_mpe = pd.DataFrame(mean_percentage_error_from_samples)

    #PMF BASED MODEL
    #evaluation of the pmf based model
    # estimation_func = AlkalinityEstimationFunction(checkpoint=os.path.join(os.path.dirname(_file_), "models", "reinjecao", "AlkalinityNetwork.ckpt")).to("cuda")
    # estimation_func.eval()

    # pmf_model_prediction = {}
    # for i in range(train_split_size, train_split_size + test_split_size):  #takes only the test samples
    #     prediction = estimation_func(calibrated_pmf = torch.as_tensor(Y_test_pmf_model[i].calibrated_pmf, dtype = torch.float32, device = "cuda"))
    #     pmf_model_prediction[f"sample_{i}"] =  prediction




if __name__ == "__main__":
    main()

