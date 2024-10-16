import torch
import os
import numpy as np
import json, yaml

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

# reduces mat mul precision (for performance)
torch.set_float32_matmul_precision('high')

### Variables ###
# reads setting`s yaml
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    # global variables
    ANALYTE = settings["analyte"]
    SKIP_BLANK = settings["skip_blank"]
    MODEL_VERSION = settings["network_model"]
    USE_CHECKPOINT = settings["use_checkpoint"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    CNN_BLOCKS = settings["cnn_blocks"]
    # training hyperparams
    MAX_EPOCHS = settings["models"]["max_epochs"]
    LR = settings["models"]["learning_rate"]
    LR_PATIENCE = settings["models"]["learning_rate_patience"]
    EARLY_STOP_PATIENCE = 2*LR_PATIENCE + 1
    LOSS_FUNCTION = settings["models"]["loss_function"]
    GRADIENT_CLIPPING = settings["models"]["gradient_clipping"]
    BATCH_SIZE = 500#settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["cnn1_output_shape"]**2   # uses all the descriptors from an single image as a batch
    # data variables
    DESCRIPTOR_DEPTH = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["descriptor_depth"]

networks_choices = {"Alkalinity":{"model_1": alkalinity.Model_1,
                                  "model_2": alkalinity.Model_2},
                    "Chloride": {"model_1": chloride.Model_1,
                                 "model_2": chloride.Model_2,
                                 "model_3": chloride.Model_3}}
MODEL_NETWORK = chloride.Model_3 #networks_choices[ANALYTE][MODEL_VERSION]

loss_function_choices = {"mean_squared_error": torch.nn.MSELoss()}
LOSS_FUNCTION = loss_function_choices[LOSS_FUNCTION]

# defines path dir
if SKIP_BLANK:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank",  "train" ))
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")

else:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank",  "train"))
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")

# creates directories
os.makedirs(CHECKPOINT_ROOT, exist_ok =True)
os.makedirs(LEARNING_VALUES_ROOT, exist_ok =True)

if USE_CHECKPOINT:
    CHECKPOINT_FILENAME = f"{MODEL_VERSION}({CNN_BLOCKS}_blocks).ckpt"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FILENAME)#f"{MODEL_VERSION}.ckpt")

else:
    os.makedirs(os.path.join(CHECKPOINT_ROOT), exist_ok =True)

### Main ###
def main():

    # define checkpoint path and monitor
    checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_ROOT, filename=f"{MODEL_VERSION}({CNN_BLOCKS}_blocks)", save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=True)#every_n_epochs=CHECKPOINT_SAVE_INTERVAL)

    # load data module
    data_module = DataModule(descriptor_root=DESCRIPTORS_ROOT, batch_size= BATCH_SIZE, num_workers=6)

    if USE_CHECKPOINT:
        model = BaseModel.load_from_checkpoint(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=BATCH_SIZE, learning_rate=LR,  learning_rate_patience=10, checkpoint_path=CHECKPOINT_PATH, descriptor_depth = DESCRIPTOR_DEPTH, sweep_config = None)

    else:
        model = BaseModel(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=BATCH_SIZE, learning_rate=LR, learning_rate_patience=10, descriptor_depth = DESCRIPTOR_DEPTH, sweep_config = None)

    # train the model
    trainer = Trainer(
                      accelerator="cuda",
                      max_epochs=MAX_EPOCHS,
                      callbacks=checkpoint_callback,
                      gradient_clip_val= GRADIENT_CLIPPING,
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
