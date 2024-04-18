import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import chemical_analysis
import shutil

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
EPOCHS = 5
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
PMF_SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", f"{ANALYTE}_Samples" )
TEST_DIR_PATH = os.makedirs(os.path.join(os.path.dirname(__file__), "..", "test_samples", f"{ANALYTE}_Samples"), exist_ok =True)
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
del X_train_descriptorss_model
del y_train_descriptorss_model
del X_test_descriptorss_model
del y_test_descriptorss_model

torch.cuda.empty_cache()


# loads data and splits into training and testing for the image pmf based model
for test_samples_count in range(train_split_size, train_split_size + test_split_size):
    shutil.copy(f"sample_{test_samples_count}", TEST_DIR_PATH)
    
Y_test_pmf_model = AlkalinitySampleDataset(
                                            base_dirs= TEST_DIR_PATH,  #TODO colocar as imagens de alkalinidade no formato ordenado de samples (utils/listed_images_samples.py)
                                            progress_bar = True, 
                                            skip_blank_samples = True, 
                                            skip_incomplete_samples = True, 
                                            skip_inference_sample= True, 
                                            skip_training_sample = False, 
                                            verbose = True
                                            ) 

Y_test_pmf_model = ProcessedAlkalinitySampleDataset(
                                                    dataset = Y_test_pmf_model, 
                                                    cache_dir = CACHE_PATH,
                                                    num_augmented_samples = 0, 
                                                    progress_bar = True, 
                                                    transform = None,
                                                    )

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


# class Statistics():

#     def __init__(self, sample_predictions_vector):
#         self.sample_predictions_vector = torch.tensor(sample_predictions_vector)
#         self.mean = torch.mean(self.sample_predictions_vector).item()
#         self.median = torch.median(self.sample_predictions_vector).item()
#         self.mode = torch.mode(torch.flatten(self.sample_predictions_vector))[0].item()
#         self.variance = torch.var(self.sample_predictions_vector).item()
#         self.std = torch.std(self.sample_predictions_vector).item()
#         self.min_value = torch.min(self.sample_predictions_vector).item()
#         self.max_value = torch.max(self.sample_predictions_vector).item()

def main():
    #training the descriptor based model
    print("Training time")
    for actual_epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)

        print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

    #evaluation of the descriptor based model
    print("Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model,eval_loader, loss_fn)

    #evaluation of the pmf based model
    estimation_func = AlkalinityEstimationFunction(checkpoint=os.path.join(os.path.dirname(_file_), "models", "reinjecao", "AlkalinityNetwork.ckpt")).to("cuda")
    estimation_func.eval()

    pmf_model_prediction = []
    for sample in Y_test_pmf_model:
        prediction = estimation_func(calibrated_pmf = torch.as_tensor(sample.calibrated_pmf, dtype = torch.float32, device = "cuda"))
        pmf_model_prediction.append(prediction)

        #TODO colocar erro absoluto e erro relativo para ambos modelos