import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math
import json

from tqdm import tqdm

from models import alkalinity, chloride
from models.lightning import DataModule, BaseModel

from torch.utils.data import DataLoader, TensorDataset
from torch import FloatTensor, UntypedStorage

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Tuple, List
if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
ANALYTE = "Chloride"
SKIP_BLANK = False
IMAGES_TO_EVALUATE = "test"
CHECKPOINT_FILENAME = "Model_4.ckpt"

if ANALYTE == "Alkalinity":
    EPOCHS = 1
    LR = 0.001
    LOSS_FUNCTION = torch.nn.MSELoss()
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = "Model_1"
    MODEL_NETWORK = alkalinity.Model_1()
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    RECEPTIVE_FIELD_DIM = 15
    IMAGE_SHAPE = 97
    IMAGE_SIZE = IMAGE_SHAPE * IMAGE_SHAPE # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    DESCRIPTOR_DEPTH = 448

elif ANALYTE == "Chloride":
    EPOCHS = 1
    LR = 0.001
    LOSS_FUNCTION = torch.nn.MSELoss()
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = "Model_4"
    MODEL_NETWORK = chloride.Model_4()
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = False
    RECEPTIVE_FIELD_DIM = 27
    IMAGE_SHAPE = 86
    IMAGE_SIZE = IMAGE_SHAPE * IMAGE_SHAPE  # after the crop based on the receptive field  (shape = (112 - 27, 112 - 27))
    DESCRIPTOR_DEPTH = 1472

if ANALYTE == "Phosphate":
    EPOCHS = 1
    LR = 0.001
    LOSS_FUNCTION = torch.nn.MSELoss()
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    #MODEL_VERSION = "Model_1"
    #MODEL_NETWORK = alkalinity.Model_1
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    #RECEPTIVE_FIELD_DIM = 15
    #IMAGE_SIZE = 97 * 97  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    #DESCRIPTOR_DEPTH = 448

if ANALYTE == "Sulfate":
    EPOCHS = 1
    LR = 0.001
    LOSS_FUNCTION = torch.nn.MSELoss()
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    #MODEL_VERSION = "Model_1"
    #MODEL_NETWORK = alkalinity.Model_1
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    #RECEPTIVE_FIELD_DIM = 15
    #IMAGE_SIZE = 97 * 97  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    #DESCRIPTOR_DEPTH = 448

if SKIP_BLANK:
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank"))
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}",  "no_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "Udescriptors","no_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", f"{IMAGES_TO_EVALUATE}")
else:
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank"))
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "with_blank")
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "Udescriptors", "with_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", f"{IMAGES_TO_EVALUATE}")

CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FILENAME)
print('Using this checkpoint:', CHECKPOINT_PATH)

EXPECTED_RANGE = {
                "Alkalinity": (500.0, 2500.0),
                "Chloride": (10000.0, 300000.0),
                "Phosphate": (0.0, 50.0),
                "Sulfate":(0.0, 4000.0),
                 }



#creates directories
os.makedirs(EVALUATION_ROOT, exist_ok =True)

os.makedirs(os.path.join(EVALUATION_ROOT, "train", "predicted_values", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, "train", "histogram", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, "train", "error_from_image", "from_cnn1_output", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, "train", "error_from_image", "from_original_image", MODEL_VERSION), exist_ok =True)

os.makedirs(os.path.join(EVALUATION_ROOT, "test", "predicted_values", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, "test", "histogram", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, "test", "error_from_image", "from_cnn1_output", MODEL_VERSION), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, "test", "error_from_image", "from_original_image", MODEL_VERSION), exist_ok =True)

model = MODEL_NETWORK.to('cuda')
loss_fn = LOSS_FUNCTION#torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_states'][0])

# utilities functions and classes

# loads datasets for evaluation
def load_dataset(dataset_for_inference: str, descriptor_root: str = DESCRIPTORS_ROOT):
        with open(os.path.join(descriptor_root, f'metadata_{dataset_for_inference}.json'), "r") as file:
            metadata = json.load(file)
        total_samples = metadata['total_samples']
        image_size = metadata['image_size']
        descriptor_depth = metadata['descriptor_depth']
        nbytes_float32 = torch.finfo(torch.float32).bits//8

        #NOTE:
        # at the moment, descriptors are saved in the format (num samples, image_size, descriptors_depth), but they are read in format (num samples * image_size,descriptors_depth).
        # expected_value is saved in format (num samples, image_size), and read in format (num samples * image_size)
        descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(descriptor_root, f"descriptors_{dataset_for_inference}.bin"), shared = False, nbytes= (total_samples * image_size * descriptor_depth) * nbytes_float32)).view(total_samples * image_size, descriptor_depth)
        expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(descriptor_root, f"descriptors_anotation_{dataset_for_inference}.bin"), shared = False, nbytes= (total_samples * image_size) * nbytes_float32)).view(total_samples * image_size)

        return TensorDataset(descriptors.to("cuda"), expected_value.to("cuda"))

# train the model
# def train_epoch(
#                 model: torch.nn,
#                 train_loader: TensorDataset,
#                 optimizer: torch.optim ,
#                 loss_fn: torch.nn = loss_fn) -> float:

#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         y_pred = model(X_batch.unsqueeze(0)).squeeze(1)
#         loss = loss_fn(y_pred, y_batch)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING_VALUE)
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(train_loader)

#evaluates the model
def evaluate(
            model: torch.nn,
            eval_loader: TensorDataset,
            loss_fn: torch.nn = loss_fn) -> Tuple[np.array, np.array, np.array]:

    model.eval()  # change model to evaluation mode

    partial_loss = []
    predicted_value = []
    expected_value = []
    #total_samples = len(eval_loader)

    with torch.no_grad():
        for X_batch, y_batch in eval_loader:

            y_pred = model(X_batch.unsqueeze(0))
            y_pred = y_pred.squeeze()
            predicted_value.append(round(y_pred.item(), 2))

            expected_value.append(y_batch.item())

            loss = loss_fn(y_pred, y_batch)
            partial_loss.append(loss.item())

    partial_loss = np.array(partial_loss)
    predicted_value = np.array(predicted_value)
    expected_value = np.array(expected_value)

    return partial_loss, predicted_value, expected_value # ,accuracy

def get_min_max_values(dataset_for_inference):

    if dataset_for_inference == 'train':
        descriptor_root = os.path.join(DESCRIPTORS_ROOT, "train")

    elif dataset_for_inference == 'test':
        descriptor_root = os.path.join(DESCRIPTORS_ROOT, "test")

    else:
        raise NotImplementedError

    with open(os.path.join(descriptor_root, f'metadata_{dataset_for_inference}.json'), "r") as file:
            metadata = json.load(file)
            total_samples = metadata['total_samples']
            image_size = metadata['image_size']
            descriptor_depth = metadata['descriptor_depth']
            nbytes_float32 = torch.finfo(torch.float32).bits//8


    # reads the untyped storage object of saved descriptors
    #descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(descriptors_path, "descriptors.bin"), shared=False, nbytes=(dim * DESCRIPTOR_DEPTH) * torch.finfo(torch.float32).bits // 8)).view(IMAGE_SIZE * total_samples, DESCRIPTOR_DEPTH)
    expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(descriptor_root, f"descriptors_anotation_{dataset_for_inference}.bin"), shared = False, nbytes= (total_samples * image_size) * nbytes_float32)).view(total_samples * image_size)

    # saves values for graph scale
    min_value = float(torch.min(expected_value[:]))
    max_value = float(torch.max(expected_value[:]))

    return min_value, max_value

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

def main(dataset_for_inference):
    #TODO alterar isso para abrir a parti do json de metadados
    train_samples_len = (int(len(os.listdir(os.path.join(SAMPLES_PATH, "train")))/3))
    test_samples_len = (int(len(os.listdir(os.path.join(SAMPLES_PATH, "test")))/3))

    if dataset_for_inference == "train":
        len_mode = train_samples_len
        dataset = load_dataset("train")

        save_histogram_path = os.path.join(EVALUATION_ROOT, "train", "histogram", MODEL_VERSION)
        save_error_from_cnn1_path = os.path.join(EVALUATION_ROOT, "train", "error_from_image", "from_cnn1_output", MODEL_VERSION)
        save_error_from_image_path = os.path.join(EVALUATION_ROOT, "train", "error_from_image","from_original_image", MODEL_VERSION)
        original_image_path = os.path.join(ORIGINAL_IMAGE_ROOT, "train")

    elif dataset_for_inference == "test":
         len_mode = test_samples_len
         dataset = load_dataset("test")

         save_histogram_path = os.path.join(EVALUATION_ROOT, "test", "histogram", MODEL_VERSION, )
         save_error_from_cnn1_path = os.path.join(EVALUATION_ROOT, "test", "error_from_image", "from_cnn1_output", MODEL_VERSION)
         save_error_from_image_path = os.path.join(EVALUATION_ROOT,  "test", "error_from_image", "from_original_image", MODEL_VERSION)
         original_image_path = os.path.join(ORIGINAL_IMAGE_ROOT, "test")

    else:
        NotImplementedError

    #training time
    # print("Training time")
    # for actual_epoch in tqdm(range(EPOCHS)):
    #     train_loss = train_epoch(model=model, train_loader=dataset, optimizer=optimizer, loss_fn=loss_fn)

    #     print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

    # gets the model used
    #model = base_model.model.to('cuda')

    #evaluation time
    print("Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model=model, eval_loader=dataset)

    #transforms data before saving
    values_ziped = zip(predicted_value, expected_value)  #zips predicted and expected values
    column_array_values = np.array(list(values_ziped))  # converts to numpy
    #saves prediction`s data
    with open(os.path.join(EVALUATION_ROOT, dataset_for_inference, "predicted_values", MODEL_VERSION, f"{MODEL_VERSION}.txt"), "w") as file: # overrides if file exists
        file.write("predicted_value,expected_value\n")

    with open(os.path.join(EVALUATION_ROOT, dataset_for_inference, "predicted_values", MODEL_VERSION, f"{MODEL_VERSION}.txt"), "a+") as file:
        for line in column_array_values:
            file.write(f"{line[0]}, {line[1]}\n")

    #histograms
    print("calculating histograms of predictions\n")
    predicted_value_for_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(predicted_value,( -1, IMAGE_SHAPE, IMAGE_SHAPE)))}
    expected_value_from_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(expected_value,( -1, IMAGE_SHAPE, IMAGE_SHAPE)))}

    min_value, max_value = get_min_max_values(dataset_for_inference)

    for i in range(len_mode):
        values = np.array(predicted_value_for_samples[f'sample_{i}']).flatten() #flattens for histogram calculation
        stats = Statistics(predicted_value_for_samples[f'sample_{i}'], expected_value_from_samples[f'sample_{i}'])
        plt.figure(figsize=(15, 8))

        #counts, bins = np.unique(values, return_counts = True)
        bins = int(math.ceil(max_value/(EXPECTED_RANGE[ANALYTE][0]*0.1/2))) #valor maximo do analito / metade do pior erro relativo (10% do menor valor esperado)
        plt.hist(values, bins = bins, range = (min_value, max_value), color='black')

        # adds vertical lines for basic statistic values
        plt.axvline(x = stats.mean, alpha = 0.5, c = 'red')
        plt.axvline(x = stats.median, alpha = 0.5, c = 'blue')
        #plt.axvline(x = stats.mode, alpha = 0.5, c = 'green')

        plt.title(f"Sample_{i }, Expected Value: {round(expected_value_from_samples[f'sample_{i}'][0][0], 2)}")
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
        plt.savefig(os.path.join(save_histogram_path, f"sample_{i}.png"))
        plt.close('all')

    # reshapes to the size of the output from the first cnn in vgg11  and the total of images
    print("reshaping images to match cnn1 output\n")
    partial_loss_from_cnn1_output = np.reshape(partial_loss, (-1, IMAGE_SHAPE, IMAGE_SHAPE))

    for i, image in enumerate(partial_loss_from_cnn1_output):
        plt.imsave(os.path.join(save_error_from_cnn1_path, f"sample_{i}.png"), image)

    # resize to the original input size (224,224)
    print("resizing images to original size\n")
    partial_loss_from_original_image = []#np.resize(partial_loss_from_cnn1_output, (224,224))
    for image in partial_loss_from_cnn1_output:
        resized_image = cv2.resize(image, (206, 206), interpolation = cv2.INTER_NEAREST)
        partial_loss_from_original_image.append(resized_image)

    partial_loss_from_original_image = np.array(partial_loss_from_original_image) #to optimize computing

    rf = int((RECEPTIVE_FIELD_DIM - 1)/2)  #  alkalinity:  (15-1)/2,  chloride :   (27-1)/2
    for i in range(len(partial_loss_from_original_image) -1):
        original_image = cv2.imread(os.path.join(original_image_path, f"sample_{i}.png"))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = original_image[rf : original_image.shape[0] - rf,  rf : original_image.shape[1] - rf,:] # cropps the image to match the cropped image after 3rd cnn

        #plots error map and original image sidewise
        fig,ax = plt.subplots(nrows=1,ncols=2)
        fig.suptitle(f"Sample_{i}: error map  X  original image")
        ax[0].imshow(partial_loss_from_original_image[i])
        ax[1].imshow(original_image)
        plt.savefig(os.path.join(save_error_from_image_path, f"sample_{i}.png"))
        plt.close('all')

        #plt.imsave(f"./evaluation/error_from_image/from_original_image/{MODEL_VERSION}/sample_{i}.png", image)

    write_pdf_statistics()


if __name__ == "__main__":
    #main(dataset_for_inference="train")
    main(dataset_for_inference="test")