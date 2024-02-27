import torch
import os
import numpy as np
import matplotlib.pyplot as plt 
import cv2

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# variables
EPOCHS = 5
LR = 0.0001
BATCH_SIZE = 1024  
EVALUATION_BATCH_SIZE = 1
GRADIENT_CLIPPING_VALUE = 0.5
MODEL_VERSION = 'model_1' #if len(os.listdir("./models")) == 0 else f'model_{len(os.listdir("./models"))}'
take_last_checkpoint = sorted(os.listdir(f"./checkpoints/{MODEL_VERSION}"), key = lambda x: int(x.split('_')[-1]))[-1]
CHECKPOINT_PATH = f"./checkpoints/{MODEL_VERSION}/{take_last_checkpoint}"  #CHECKPOINT_PATH = f"./checkpoints/{MODEL_VERSION}/{MODEL_VERSION}_epoch_3000"
print('Using this checkpoint:', CHECKPOINT_PATH)

#creates directories
os.makedirs("./evaluation", exist_ok =True)
os.makedirs(f"./evaluation/predicted_values/{MODEL_VERSION}", exist_ok =True)
os.makedirs(f"./evaluation/histogram/{MODEL_VERSION}", exist_ok =True)
os.makedirs(f"./evaluation/error_from_image/from_cnn1_output/{MODEL_VERSION}", exist_ok =True)
os.makedirs(f"./evaluation/error_from_image/from_original_image/{MODEL_VERSION}", exist_ok =True)

# loads data and splits into training and testing
X = torch.cat([torch.load(f"../descriptors/sample_{i}") for i in range(int(len(os.listdir("../descriptors")) / 2))], dim=0)
y = torch.cat([torch.load(f"../descriptors/sample_{i}_anotation") for i in range(int(len(os.listdir("../descriptors")) / 2))], dim=0)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X, y

print(f"X size: {X.size()}")
print(f"y size: {y.size()}")

# makes batchers for training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size = BATCH_SIZE, shuffle= True)
eval_loader = DataLoader(list(zip(X_train, y_train)), batch_size = EVALUATION_BATCH_SIZE, shuffle = False)

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
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


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
for actual_epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    
    print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

#evaluation
print("Evaluation time")
_, predicted_value, expected_value = evaluate(model,eval_loader, loss_fn)


with open(f"./evaluation/predicted_values/{MODEL_VERSION}/{MODEL_VERSION}.txt", "w") as file: # overrides if file exists
    file.write("predicted_value,expected_value\n")
    
with open(f"./evaluation/predicted_values/{MODEL_VERSION}/{MODEL_VERSION}.txt", "a+") as file:
    for i in range(len(predicted_value)):
        file.write(f"{predicted_value[i]},{expected_value[i]}\n")

#histograms
predicted_value_for_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(predicted_value,( -1, 112,112)))}
expected_value_from_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(expected_value,( -1, 112,112)))}
for i in range(int(len(eval_loader)/(112*112))):
    values = predicted_value_for_samples[f'sample_{i}']
    plt.hist(values, edgecolor ='black')
    plt.title(f"Sample_{i}, Expected Value: {round(expected_value_from_samples[f'sample_{i}'][0][0], 2)}")
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.savefig(f'./evaluation/histogram/{MODEL_VERSION}/sample_{i}.png')

# reshapes to the size of the output from the first cnn in vgg11 (112,112) and the total of images (len(eval_loader)/(112*112) = 695)
partial_loss_from_cnn1_output = np.reshape(partial_loss, -1, 112,112)

for i, image in enumerate(partial_loss_from_cnn1_output):
    plt.imsave(f"./evaluation/error_from_image/from_cnn1_output/{MODEL_VERSION}/sample_{i}.png", image)


# resize to the original input size (224,224)
partial_loss_from_original_image = []#np.resize(partial_loss_from_cnn1_output, (224,224))
for image in partial_loss_from_cnn1_output:
    resized_image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_NEAREST)
    partial_loss_from_original_image.append(resized_image)

for i, image in enumerate(partial_loss_from_original_image):
    plt.imsave(f"./evaluation/error_from_image/from_original_image/{MODEL_VERSION}/sample_{i}.png", image)