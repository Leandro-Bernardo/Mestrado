import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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

# clears data from memory
del X_train
del y_train
del X_test
del y_test

torch.cuda.empty_cache()

#model definition
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

# utilities functions
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
    values = predicted_value_for_samples[f'sample_{i}']
    plt.hist(values, edgecolor ='black')
    plt.title(f"Sample_{i}, Expected Value: {round(expected_value_from_samples[f'sample_{i}'][0][0], 2)}")
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.savefig(os.path.join(EVALUATION_ROOT, "histogram", MODEL_VERSION, f"sample_{i + train_split_size + 1}.png"))
    plt.close('all')

# reshapes to the size of the output from the first cnn in vgg11 (112 - 18, 112 - 18) and the total of images (len(eval_loader)/(112*112) = 695)
print("reshaping images to match cnn1 output\n")
partial_loss_from_cnn1_output = np.reshape(partial_loss, -1, 94, 94)

for i, image in enumerate(partial_loss_from_cnn1_output):
    plt.imsave(os.path.join(EVALUATION_ROOT, "error_from_image","from_cnn1_output", MODEL_VERSION, f"sample_{i + train_split_size + 1}.png"), image)


# resize to the original input size (224,224)
print("resizing images to original size\n")
partial_loss_from_original_image = []#np.resize(partial_loss_from_cnn1_output, (224,224))
for image in partial_loss_from_cnn1_output:
    resized_image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_NEAREST)
    partial_loss_from_original_image.append(resized_image)

for i, error_map in enumerate(partial_loss_from_original_image):
    original_image = cv2.cvtColor(cv2.imread(f"./images/sample_{i + train_split_size + 1}.png"), cv2.COLOR_BGR2RGB)

    #plots error map and original image sidewise
    fig,ax = plt.subplots(nrows=1,ncols=2)
    fig.suptitle(f"Sample_{i + train_split_size + 1}: error map  X  original image")
    ax[0].imshow(error_map)
    ax[1].imshow(original_image)
    plt.savefig(os.path.join(EVALUATION_ROOT, "error_from_image", "from_original_image", MODEL_VERSION, f"sample_{i + train_split_size + 1}.png"))
    plt.close('all')

    #plt.imsave(f"./evaluation/error_from_image/from_original_image/{MODEL_VERSION}/sample_{i}.png", image)