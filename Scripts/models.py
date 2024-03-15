import torch
import os

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
FIRST_EPOCH = 0
FINAL_EPOCH = 5000
LR = 0.0001
BATCH_SIZE = 64  
GRADIENT_CLIPPING_VALUE = 0.5
CHECKPOINT_SAVE_INTERVAL = 25
MODEL_VERSION = 'model_1' 
DATASET_SPLIT = 0.8
USE_CHECKPOINT = False

#defines path dir
CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints")
DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "descriptors")
LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values")


# creates directories
os.makedirs(CHECKPOINT_ROOT, exist_ok =True)
os.makedirs(LEARNING_VALUES_ROOT, exist_ok =True)

if USE_CHECKPOINT:
    LAST_CHECKPOINT = sorted(os.listdir(os.path.join(CHECKPOINT_ROOT, MODEL_VERSION)), key = lambda x: int(x.split('_')[-1]))
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, MODEL_VERSION, LAST_CHECKPOINT[-1])
    print('Using this checkpoint:', CHECKPOINT_PATH)
    FIRST_EPOCH = int(CHECKPOINT_PATH.split('_')[-1]) + 1
else:
    os.makedirs(os.path.join(CHECKPOINT_ROOT, MODEL_VERSION), exist_ok =True)

# loads data and splits into training and testing
list_files = os.listdir(DESCRIPTORS_ROOT)
files_size = len(list_files)
train_split_size = int((files_size // 2) * DATASET_SPLIT)
test_split_size = int((files_size // 2) - train_split_size)


X_train = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}")) for i in range(train_split_size)], dim=0).to(device=device)
y_train = torch.cat([torch.load(os.path.join(DESCRIPTORS_ROOT, f"sample_{i}_anotation")) for i in range(train_split_size)], dim=0).to(device=device)

print(f"X_train, y_train size: {X_train.size()}, {y_train.size()}")

# makes batchers for training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size = BATCH_SIZE, shuffle= True)

del X_train
del y_train

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

if USE_CHECKPOINT:
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

def checkpoint(model_name, model, optimizer, epoch): #saves the models params
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"./checkpoints/{model_name}/{model_name}_epoch_{epoch}")


#training   
if USE_CHECKPOINT:
    pass
else:
    with open(os.path.join(LEARNING_VALUES_ROOT, f"{MODEL_VERSION}.txt", 'w')) as file:
        file.write("Epoch\tLoss\n")
       
for epoch in tdqm(range(FIRST_EPOCH, FINAL_EPOCH)):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    
    with open(os.path.join(LEARNING_VALUES_ROOT, f"{MODEL_VERSION}.txt", 'a+')) as file:
        file.write(f"{epoch}\t{train_loss}\n")
    
    print(f"Epoch {epoch}, Loss: {train_loss}")

    if (epoch) % CHECKPOINT_SAVE_INTERVAL == 0:  #saves model weights at fixed rate
        checkpoint(MODEL_VERSION, model, optimizer, epoch)
        