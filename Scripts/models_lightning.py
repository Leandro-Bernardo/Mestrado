import torch
import os
import numpy as np

from tqdm import tqdm
from models import alkalinity, chloride
from models.lightning import DataModule, BaseModel

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if not torch.cuda.is_available():
    assert("cuda isnt available")
    device = "cuda"

else:
    device = "cuda"

#Variables
ANALYTE = "Chloride"
SKIP_BLANK = False
USE_CHECKPOINT = True

if ANALYTE == "Alkalinity":
    MODEL_VERSION = "Model_2"
    MODEL_NETWORK = alkalinity.Model_2
    FIRST_EPOCH = 0
    FINAL_EPOCH = 5000
    LR = 0.001
    LOSS_FUNCTION = torch.nn.MSELoss()
    GRADIENT_CLIPPING_VALUE = 0.5
    #CHECKPOINT_SAVE_INTERVAL = 25
    IMAGE_SIZE = 97 * 97  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    BATCH_SIZE = IMAGE_SIZE
    DESCRIPTOR_DEPTH = 448


elif ANALYTE == "Chloride":
    MODEL_VERSION = "Model_1"
    MODEL_NETWORK = chloride.Model_1
    FIRST_EPOCH = 0
    FINAL_EPOCH = 5000
    LR = 0.001
    LOSS_FUNCTION = torch.nn.MSELoss()
    GRADIENT_CLIPPING_VALUE = 0.5
    #CHECKPOINT_SAVE_INTERVAL = 20
    IMAGE_SIZE = 86 * 86  # after the crop based on the receptive field  (shape = (112 - 27, 112 - 27))
    BATCH_SIZE = IMAGE_SIZE
    DESCRIPTOR_DEPTH = 1472

#defines path dir
if SKIP_BLANK:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank")
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank",  "train" ))
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "no_blank")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "no_blank")

else:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank")
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank",  "train"))
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "with_blank")


# creates directories
os.makedirs(CHECKPOINT_ROOT, exist_ok =True)
os.makedirs(LEARNING_VALUES_ROOT, exist_ok =True)

if USE_CHECKPOINT:
    CHECKPOINT_FILENAME = f"{MODEL_VERSION}.ckpt"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FILENAME)#f"{MODEL_VERSION}.ckpt")

else:
    os.makedirs(os.path.join(CHECKPOINT_ROOT), exist_ok =True)

def main():

    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_ROOT, filename=f"{MODEL_VERSION}", save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=True)#every_n_epochs=CHECKPOINT_SAVE_INTERVAL)

    #train_dataset = PrepareDataset(descriptors_root= DESCRIPTORS_ROOT, stage="train")
    #val_dataset = PrepareDataset(descriptors_root= DESCRIPTORS_ROOT, stage="val")

    data_module = DataModule(descriptor_root=DESCRIPTORS_ROOT, stage="train", train_batch_size= BATCH_SIZE, num_workers=2)

    if USE_CHECKPOINT:
        model = BaseModel.load_from_checkpoint(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=BATCH_SIZE, learning_rate=LR,  learning_rate_patience=10, checkpoint_path=CHECKPOINT_PATH)

    else:
        model = BaseModel(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=BATCH_SIZE, learning_rate=LR, learning_rate_patience=10)

    #trains the model
    trainer = Trainer(
                      accelerator="cuda",
                      max_epochs=FINAL_EPOCH,
                      callbacks=checkpoint_callback,
                      gradient_clip_val=0.5,
                      gradient_clip_algorithm="value",  # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping
                      log_every_n_steps=1,
                      num_sanity_val_steps=0,
                      enable_progress_bar=True
                    )

    trainer.fit(model=model, datamodule=data_module)#, train_dataloaders=dataset


if __name__ == "__main__":
    if USE_CHECKPOINT:
        print(f"Using this checkpoint: {CHECKPOINT_PATH}")
    main()
