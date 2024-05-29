import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
ANALYTE = "Alkalinity"
SKIP_BLANK = True

if ANALYTE == "Alkalinity":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    RECEPTIVE_FIELD_DIM = 15

elif ANALYTE == "Chloride":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_3'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = False
    RECEPTIVE_FIELD_DIM = 27

if ANALYTE == "Phosphate":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 64
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True

if ANALYTE == "Sulfate":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 64
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True

if SKIP_BLANK:
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "no_blank")
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "no_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "no_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank")
else:
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "with_blank")
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "with_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "with_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank")

LAST_CHECKPOINT = sorted(os.listdir(os.path.join(CHECKPOINT_ROOT, MODEL_VERSION)), key = lambda x: int(x.split('_')[-1]))[-1]
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, MODEL_VERSION, LAST_CHECKPOINT)

EXPECTED_RANGE = {
                "Alkalinity": (500.0, 2500.0),
                "Chloride": (10000.0, 300000.0),
                "Phosphate": (0.0, 50.0),
                "Sulfate":(0.0, 4000.0),
                 }

print('Using this checkpoint:', CHECKPOINT_PATH)

#creates directories
os.makedirs(EVALUATION_ROOT, exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT,"predicted_values", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT,"histogram", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT,"error_from_image", "from_cnn1_output", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT,"error_from_image", "from_original_image", MODEL_VERSION), exist_ok =True)

# loads data and splits into training and testing
list_files = os.listdir(DESCRIPTORS_ROOT)
files_size = len(list_files)
train_split_size = int((files_size // 2) * DATASET_SPLIT)
test_split_size = int((files_size // 2) - train_split_size)

X_train = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size)], dim=0).to(device=device)
y_train = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size)], dim=0).to(device=device)

X_test = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size, train_split_size + test_split_size)], dim=0).to(device=device)
y_test = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size, train_split_size + test_split_size)], dim=0).to(device=device)

print(f"X_train, y_train size: {X_train.size()}, {y_train.size()}")
print(f"X_test, y_test size: {X_test.size()}, {y_test.size()}")

# makes batchers for training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size = BATCH_SIZE, shuffle= True)
eval_loader = DataLoader(list(zip(X_test, y_test)), batch_size = EVALUATION_BATCH_SIZE, shuffle = False)

# saves values for graph scale
min_value, max_value = float(torch.min(y_test)), float(torch.max(y_test))

# clears data from memory
del X_train
del y_train
del X_test
del y_test

torch.cuda.empty_cache()

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

def write_pdf_statistics():
    pass

def main():
    #training
    print("Training time")
    for actual_epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)

        print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

    #evaluation
    print("Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model,eval_loader, loss_fn)


    with open(os.path.join(EVALUATION_ROOT, "predicted_values", MODEL_VERSION, f"{MODEL_VERSION}.txt"), "w") as file: # overrides if file exists
        file.write("predicted_value,expected_value\n")

    with open(os.path.join(EVALUATION_ROOT, "predicted_values", MODEL_VERSION, f"{MODEL_VERSION}.txt"), "a+") as file:
        for i in range(len(predicted_value)):
            file.write(f"{predicted_value[i]},{expected_value[i]}\n")

    #histograms
    print("calculating histograms of predictions\n")
    predicted_value_for_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(predicted_value,( -1, 94, 94)))}
    expected_value_from_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(expected_value,( -1, 94, 94)))}

    for i in range(int(len(eval_loader)/(94*94))):
        values = np.array(predicted_value_for_samples[f'sample_{i}']).flatten() #flattens for histogram calculation
        stats = Statistics(predicted_value_for_samples[f'sample_{i}'], expected_value_from_samples[f'sample_{i}'])
        plt.figure(figsize=(15, 8))

        #counts, bins = np.unique(values, return_counts = True)
        bins = int(math.ceil(max_value/(EXPECTED_RANGE[ANALYTE][0]*0.1/2))) #valor maximo do analito / metade do pior erro relativo (10% do menor valor esperado)
        plt.hist(values, bins = bins, range = (min_value, max_value), color='black')  #bins = bins ,color='black') #bins=len(bins), color='black')

        # adds vertical lines for basic statistic values
        plt.axvline(x = stats.mean, alpha = 0.5, c = 'red')
        plt.axvline(x = stats.median, alpha = 0.5, c = 'blue')
        #plt.axvline(x = stats.mode, alpha = 0.5, c = 'green')

        plt.title(f"Sample_{i + train_split_size}, Expected Value: {round(expected_value_from_samples[f'sample_{i}'][0][0], 2)}")
        plt.xlabel('Prediction')
        plt.ylabel('Count')

        # defines x axis limits
        x_min, x_max = min_value, max_value
        plt.xlim((x_min, x_max))

        # get the limits from y axis (which is count based)
        y_min, y_max = plt.ylim()

        # Defines a scale factor for positions in y axis
        scale_factor = 0.08 * y_max

        #TODO FIX THIS LATER
        if ANALYTE == "Alkalinity":
            text = 210
            text_median = 240
        if ANALYTE == "Chloride":
            text = 10000
            text_median = 12000
        #text settings
        plt.text(x = x_max+0.5,  y=y_max - 10.20 * scale_factor,
                s = f" mean:")

        plt.text(x = x_max + text, y=y_max - 10.20 * scale_factor,
                s = f" {stats.mean:.2f}", c = 'red', alpha = 0.6)

        plt.text(x = x_max ,  y=y_max - 10.49 * scale_factor,
                s = f" median:" )

        plt.text(x = x_max + text_median, y=y_max - 10.49 * scale_factor,
                s = f"  {stats.median:.2f}", c = 'blue', alpha = 0.6)

        plt.text(x = x_max ,  y=y_max - 10.78 * scale_factor,
                s = f" mode:" )

        plt.text(x = x_max + text,  y=y_max - 10.78 * scale_factor,
                s = f" {stats.mode:.2f}", c = 'black', alpha = 0.6)

        plt.text(x = x_max ,  y=y_max - 12.4 * scale_factor,
                s = f" var: {stats.variance:.2f}\n std: {stats.std:.2f}\n mad: {stats.mad:.2f}\n min: {stats.min_value}\n max: {stats.max_value:.2f}", c = 'black')

        #plt.text(x = max_value + 0.5,  y = 0,
        #         s = f" mean: {stats.mean:.2f}\n median: {stats.median:.2f}\n mode: {stats.mode:.2f}\n var: {stats.variance:.2f}\n std: {stats.std:.2f}\n min: {stats.min_value:.2f}\n max: {stats.max_value:.2f}")
        plt.savefig(os.path.join(EVALUATION_ROOT, "histogram", MODEL_VERSION, f"sample_{i + train_split_size + 1}.png"))
        plt.close('all')

    # reshapes to the size of the output from the first cnn in vgg11 (112 - 18, 112 - 18) and the total of images (len(eval_loader)/(112*112) = 695)
    print("reshaping images to match cnn1 output\n")
    partial_loss_from_cnn1_output = np.reshape(partial_loss, (-1, 94, 94))

    for i, image in enumerate(partial_loss_from_cnn1_output):
        plt.imsave(os.path.join(EVALUATION_ROOT, "error_from_image","from_cnn1_output", MODEL_VERSION, f"sample_{i + train_split_size + 1}.png"), image)

    # resize to the original input size (224,224)
    print("resizing images to original size\n")
    partial_loss_from_original_image = []#np.resize(partial_loss_from_cnn1_output, (224,224))
    for image in partial_loss_from_cnn1_output:
        resized_image = cv2.resize(image, (206, 206), interpolation = cv2.INTER_NEAREST)
        partial_loss_from_original_image.append(resized_image)

    partial_loss_from_original_image = np.array(partial_loss_from_original_image) #to optimize computing

    rf = int((RECEPTIVE_FIELD_DIM - 1)/2)  #  alkalinity:  (15-1)/2,  chloride :   (27-1)/2
    for i in range(len(partial_loss_from_original_image) -1):
        original_image = cv2.imread(os.path.join(ORIGINAL_IMAGE_ROOT, f"sample_{i + train_split_size}.png"))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = original_image[rf : original_image.shape[0] - rf,  rf : original_image.shape[1] - rf,:] # cropps the image to match the cropped image after 3rd cnn

        #plots error map and original image sidewise
        fig,ax = plt.subplots(nrows=1,ncols=2)
        fig.suptitle(f"Sample_{i + train_split_size}: error map  X  original image")
        ax[0].imshow(partial_loss_from_original_image[i])
        ax[1].imshow(original_image)
        plt.savefig(os.path.join(EVALUATION_ROOT, "error_from_image", "from_original_image", MODEL_VERSION, f"sample_{i + train_split_size}.png"))
        plt.close('all')

        #plt.imsave(f"./evaluation/error_from_image/from_original_image/{MODEL_VERSION}/sample_{i}.png", image)

    write_pdf_statistics()


if __name__ == "__main__":
    main()