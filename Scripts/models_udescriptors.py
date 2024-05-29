import torch
import os
import numpy as np

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import FloatTensor, UntypedStorage

if not torch.cuda.is_available():
    assert("cuda isnt available")
    device = "cuda"

else:
    device = "cuda"

# variables
ANALYTE = "Chloride"
SKIP_BLANK = False
USE_CHECKPOINT = True

if ANALYTE == "Alkalinity":
    FIRST_EPOCH = 0
    FINAL_EPOCH = 5000
    LR = 0.001
    BATCH_SIZE = 64
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    IMAGE_SIZE = 97 * 97  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    DESCRIPTOR_DEPTH = 448


elif ANALYTE == "Chloride":
    FIRST_EPOCH = 0
    FINAL_EPOCH = 5000
    LR = 0.001
    BATCH_SIZE = 64
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 2
    MODEL_VERSION = 'model_4'
    DATASET_SPLIT = 0.8
    IMAGE_SIZE = 86 * 86  # after the crop based on the receptive field  (shape = (112 - 27, 112 - 27))
    DESCRIPTOR_DEPTH = 1472

#defines path dir
if SKIP_BLANK:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "Udescriptors", "no_blank")
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank",  "train" ))
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "no_blank", "train")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "no_blank")

else:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "Udescriptors", "with_blank")
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank",  "train"))
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank",  "train")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "with_blank")


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

#model definition

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
        #TODO aumentar camadas
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

if USE_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# utilities functions

# reads the untyped storage object of saved descriptors
TOTAL_SAMPLES = (int(len(os.listdir(SAMPLES_PATH))/3))
dim = TOTAL_SAMPLES * IMAGE_SIZE
nbytes_float32 = torch.finfo(torch.float32).bits//8
descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(DESCRIPTORS_ROOT, "descriptors.bin"), shared = True, nbytes= (dim * DESCRIPTOR_DEPTH) * nbytes_float32)).view(dim, DESCRIPTOR_DEPTH)
expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(DESCRIPTORS_ROOT, "descriptors_anotation.bin"), shared = False, nbytes= (dim) * nbytes_float32)).view(dim)

# loads data
def  load_files(starting_point, batch_size):
    X_values = descriptors[starting_point: starting_point + batch_size, :].to('cuda')
    y_values = expected_value[starting_point: starting_point + batch_size].to('cuda')

    dataset = DataLoader(list(zip(X_values, y_values)), batch_size = batch_size, shuffle= True)
    return dataset

def train_epoch(model, train_loader, optmizer, loss_fn):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze(1)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING_VALUE)
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(train_loader)

def checkpoint(model_name, model, optimizer, epoch): #saves the models params
    torch.save(
        {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        },
        os.path.join(CHECKPOINT_ROOT, model_name, f"{model_name}_epoch_{epoch}")
            )


#training
if USE_CHECKPOINT:
    pass
else:
    with open(os.path.join(LEARNING_VALUES_ROOT, f"{MODEL_VERSION}.txt"), 'w') as file:
        file.write("Epoch\tLoss\n")

for epoch in tqdm(range(FIRST_EPOCH, FINAL_EPOCH)):
    starting_point = 0
    batches_loss = 0
    num_batches = int(TOTAL_SAMPLES * IMAGE_SIZE / BATCH_SIZE)
    for _ in range(num_batches):
        sample_batch = load_files(starting_point, BATCH_SIZE)
        loss = train_epoch(model, sample_batch, optimizer, loss_fn)
        batches_loss += loss
        starting_point+= BATCH_SIZE

    mean_loss = batches_loss/num_batches

    with open(os.path.join(LEARNING_VALUES_ROOT, f"{MODEL_VERSION}.txt"), 'a+') as file:
        file.write(f"{epoch}\t{mean_loss}\n")

    print(f"  Epoch {epoch}, Loss: {mean_loss}")

    if (epoch) % CHECKPOINT_SAVE_INTERVAL == 0:  #saves model weights at fixed rate
        checkpoint(MODEL_VERSION, model, optimizer, epoch)
