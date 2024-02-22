import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader




# variables

EPOCHS = 1
LR = 0.0001
BATCH_SIZE = 1024  # X size (8705536 descriptors, 448 features (from vgg11))
GRADIENT_CLIPPING_VALUE = 0.5
MODEL_VERSION = 'model_0' if len(os.listdir("./models")) == 0 else f'model_{len(os.listdir("./models"))}'
CHECKPOINT_PATH = f"./checkpoint/{MODEL_VERSION}/{MODEL_VERSION}_epoch_2000"


# loads data and splits into training and testing
X = torch.cat([torch.load(f"../descriptors/sample_{i}") for i in range(int(len(os.listdir("../descriptors")) / 2))], dim=0)
y = torch.cat([torch.load(f"../descriptors/sample_{i}_anotation") for i in range(int(len(os.listdir("../descriptors")) / 2))], dim=0)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X, y

print(f"X size: {X.size()}")
print(f"y size: {y.size()}")

# makes batchers for training
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE)

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
#utilities

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
    }, f"./models/{model_name}/checkpoints/{model_name}_epoch_{epoch}")


