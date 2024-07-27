import torch
import os
import numpy as np
import json, yaml
import wandb

from wandb.wandb_run import Run
from tqdm import tqdm
from models import alkalinity, chloride
from models.lightning import DataModule, BaseModel

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if not torch.cuda.is_available():
    assert("cuda isnt available")
    device = "cuda"

else:
    device = "cuda"

os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".

# reduces mat mul precision (for performance)
torch.set_float32_matmul_precision('high')

### Variables ###
# reads setting`s yaml
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    ANALYTE = settings["analyte"]
    SKIP_BLANK = settings["skip_blank"]
    MODEL_VERSION = settings["network_model"]
    USE_CHECKPOINT = settings["use_checkpoint"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    CNN_BLOCKS = settings["cnn_blocks"]
    SWEEP_ID = settings["sweep_id"]

    # training hyperparams
    MAX_EPOCHS = settings["models"]["max_epochs"]
    LR = settings["models"]["learning_rate"]
    LOSS_FUNCTION = settings["models"]["loss_function"]
    GRADIENT_CLIPPING = settings["models"]["gradient_clipping"]
    BATCH_SIZE = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["image_shape"]**2   # uses all the descriptors from an single image as a batch

# reads sweep configs yaml
with open('./sweep_config.yaml') as file:
        SWEEP_CONFIGS = yaml.load(file, Loader=yaml.FullLoader)

networks_choices = {"Alkalinity": {"model_1": alkalinity.Model_1,
                                   "model_2": alkalinity.Model_2},
                      "Chloride": {"model_1": chloride.Model_1,
                                   "model_2": chloride.Model_2,
                                   "model_3": chloride.Model_3}}
MODEL_NETWORK = networks_choices[ANALYTE][MODEL_VERSION]

loss_function_choices = {"mean_squared_error": torch.nn.MSELoss()}
LOSS_FUNCTION = loss_function_choices[LOSS_FUNCTION]

# defines path dir
if SKIP_BLANK:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank",  "train" )
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")

else:
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank",  "train")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    LEARNING_VALUES_ROOT = os.path.join(os.path.dirname(__file__), "learning_values", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")

#LOG_PATH = os.path.join(os.path.dirname(__file__), "logs")

# creates directories
os.makedirs(CHECKPOINT_ROOT, exist_ok =True)
os.makedirs(LEARNING_VALUES_ROOT, exist_ok =True)
#os.makedirs(LOG_PATH, exist_ok=True)

if USE_CHECKPOINT:
    CHECKPOINT_FILENAME = f"{MODEL_VERSION}({CNN_BLOCKS}_blocks).ckpt"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FILENAME)#f"{MODEL_VERSION}.ckpt")

else:
    os.makedirs(os.path.join(CHECKPOINT_ROOT), exist_ok =True)

### Main ###
def main():
    # starts wandb
    with wandb.init(config=SWEEP_CONFIGS) as run:
        assert isinstance(run, Run)
        # initialize logger
        logger = WandbLogger(project=ANALYTE, experiment=run)
        # initialize sweep ID
        #sweep_id = wandb.sweep(SWEEP_CONFIGS, project=PROJECT_NAME)
        # initialize wandb agent
        #wandb.agent(SWEEP_ID)
        # define checkpoint path and monitor
        configs = run.config.as_dict()

        checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_ROOT, filename=f"{MODEL_VERSION}({CNN_BLOCKS}_blocks)", save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=True, save_weights_only=True)#every_n_epochs=CHECKPOINT_SAVE_INTERVAL)
        # load data module
        data_module = DataModule(descriptor_root=DESCRIPTORS_ROOT, stage="train", train_batch_size= configs["batch_size"], num_workers=2 )

        if USE_CHECKPOINT:
            model = BaseModel.load_from_checkpoint(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=configs["batch_size"], learning_rate=configs["lr"],  learning_rate_patience=10, checkpoint_path=CHECKPOINT_PATH)

        else:
            model = BaseModel(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=configs["batch_size"], learning_rate=configs["lr"], learning_rate_patience=10)

        # train the model
        trainer = Trainer(
                        logger= logger,
                        accelerator="cuda",
                        max_epochs=MAX_EPOCHS,
                        callbacks= [checkpoint_callback,
                                    LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(
                                                monitor="Loss/Val",
                                                mode="min",
                                                patience=10
                                            ),],
                        gradient_clip_val= configs["gradient_clip"],
                        gradient_clip_algorithm="value",  # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        enable_progress_bar=True
                        )

        trainer.fit(model=model, datamodule=data_module)#, train_dataloaders=dataset
        trainer.test(model, datamodule=data_module, ckpt_path=None) #ckpt_path=None takes the best model saved

        wandb.save(os.path.join(CHECKPOINT_ROOT, f"{MODEL_VERSION}({CNN_BLOCKS}_blocks).ckpt"))

if __name__ == "__main__":
    if USE_CHECKPOINT:
        print(f"Using this checkpoint: {CHECKPOINT_PATH}")
    main()
