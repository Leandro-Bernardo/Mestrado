import scipy.ndimage
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math
import json, yaml

from tqdm import tqdm

from models import alkalinity, chloride
#from models.lightning import DataModule, BaseModel

from torch.utils.data import  TensorDataset
from torch import FloatTensor, UntypedStorage

#from pytorch_lightning import Trainer
#from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Tuple, Dict

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

### Variables ###
# reads setting`s yaml
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    # global variables
    ANALYTE = settings["analyte"]
    SKIP_BLANK = settings["skip_blank"]
    MODEL_VERSION = settings["chosen_model"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    CNN_BLOCKS = settings["cnn_blocks"]
    IMG_SIZE = settings["image_size"]

    # training hyperparams variables
    MAX_EPOCHS = settings["models"]["max_epochs"]
    LR = settings["models"]["learning_rate"]
    LOSS_FUNCTION = settings["models"]["loss_function"]
    GRADIENT_CLIPPING = settings["models"]["gradient_clipping"]
    BATCH_SIZE = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["cnn1_output_shape"]**2   # uses all the descriptors from an single image as a batch
    BATCH_NORM = settings["models"]["batch_normalization"]
    # evaluation variables
    #EPOCHS = 1  #training epochs. Disabled
    EVALUATION_BATCH_SIZE = 1
    IMAGES_TO_EVALUATE = settings["statistical_analysis"]["images_to_evaluate"]
    RECEPTIVE_FIELD_DIM = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["receptive_field_dim"]
    DESCRIPTOR_DEPTH = settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["descriptor_depth"]
    CNN1_OUTPUT_SHAPE =  settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["cnn1_output_shape"]


networks_choices = {"Alkalinity":{"model_1": alkalinity.Model_1(),
                                  "model_2": alkalinity.Model_2()},
                    "Chloride": {"model_1": chloride.Model_1(),
                                 "model_2": chloride.Model_2(),
                                 "model_3": chloride.Model_3(),
                                 #"best_model_4blocks_resnet50": chloride.Best_Model_4blocks_resnet50(DESCRIPTOR_DEPTH),
                                 #"best_model_3blocks_resnet50": chloride.Best_Model_3blocks_resnet50(DESCRIPTOR_DEPTH),
                                 #"best_model_2blocks_resnet50": chloride.Best_Model_2blocks_resnet50(DESCRIPTOR_DEPTH),
                                 #"best_model_2blocks_resnet50_img_size_448": chloride.Best_Model_2blocks_resnet50_imgsize_448(DESCRIPTOR_DEPTH),
                                 "best_model_3blocks_resnet50_img_size_448": chloride.Best_Model_3blocks_resnet50_imgsize_448(DESCRIPTOR_DEPTH),
                                 #"second_best_model_3blocks_resnet50_img_size_448": chloride.Second_Best_Model_3blocks_resnet50_img_size_448(DESCRIPTOR_DEPTH),
                                 }}
MODEL_NETWORK = networks_choices[ANALYTE][MODEL_VERSION].to("cuda")

loss_function_choices = {"mean_squared_error": torch.nn.MSELoss()}
LOSS_FUNCTION = loss_function_choices[LOSS_FUNCTION]


if SKIP_BLANK:
    # model path
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)(img_size_{IMG_SIZE})")
    # data paths
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank"))
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "no_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", f"{IMAGES_TO_EVALUATE}")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}",  "no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    # results path
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}","no_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
else:
    # model path
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)(img_size_{IMG_SIZE})")
    # data paths
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank"))
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "with_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", f"{IMAGES_TO_EVALUATE}")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")
    # results path
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "with_blank", f"{FEATURE_EXTRACTOR}({CNN_BLOCKS}_blocks)")

CHECKPOINT_FILENAME = f"checkpoint.ckpt"
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

os.makedirs(os.path.join(EVALUATION_ROOT, MODEL_VERSION, IMAGES_TO_EVALUATE, "predicted_values"), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, MODEL_VERSION, IMAGES_TO_EVALUATE, "histogram"), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, MODEL_VERSION, IMAGES_TO_EVALUATE, "error_from_image", "from_cnn1_output"), exist_ok =True)
os.makedirs(os.path.join(EVALUATION_ROOT, MODEL_VERSION, IMAGES_TO_EVALUATE, "error_from_image", "from_original_image"), exist_ok =True)

### Utilities functions and classes ###

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

# fix the state dict keys and loads it
def load_state_dict(model: torch.nn.Module, checkpoint_state_dict: Dict ):
    checkpoint_state_dict = dict(checkpoint_state_dict.items())
    if "model.in_layer" in checkpoint_state_dict.keys():
        new_state_dict = {key.replace('model.', '') : value for key, value in checkpoint_state_dict.items()}
    elif "model.sequential_layers.input_layer.0.weight" in  checkpoint_state_dict.keys():
        new_state_dict = {key.replace('model.sequential_layers.', '') : value for key, value in checkpoint_state_dict.items()}
        #new_state_dict = {key.replace('.0.', '.') : value for key, value in new_state_dict.items()}
        new_state_dict = {key.replace('layer_', 'l') : value for key, value in new_state_dict.items()}


    return model.load_state_dict(new_state_dict, strict=True)

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
            loss_fn: torch.nn = LOSS_FUNCTION) -> Tuple[np.array, np.array, np.array]:

    model.eval()  # change model to evaluation mode

    partial_loss = []
    predicted_value = []
    expected_value = []
    #total_samples = len(eval_loader)

    with torch.no_grad():
        for X_batch, y_batch in eval_loader:

            if BATCH_NORM:
                y_pred = model(X_batch.unsqueeze(0))
            else:
                y_pred = model(X_batch)
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

    # if dataset_for_inference == 'train':
    #     descriptor_root = os.path.join(DESCRIPTORS_ROOT, "train")

    # elif dataset_for_inference == 'test':
    #     descriptor_root = os.path.join(DESCRIPTORS_ROOT, "test")

    # else:
    #     raise NotImplementedError

    with open(os.path.join(DESCRIPTORS_ROOT, f'metadata_{dataset_for_inference}.json'), "r") as file:
            metadata = json.load(file)
            total_samples = metadata['total_samples']
            image_size = metadata['image_size']
            descriptor_depth = metadata['descriptor_depth']
            nbytes_float32 = torch.finfo(torch.float32).bits//8


    # reads the untyped storage object of saved descriptors
    #descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(descriptors_path, "descriptors.bin"), shared=False, nbytes=(dim * DESCRIPTOR_DEPTH) * torch.finfo(torch.float32).bits // 8)).view(IMAGE_SIZE * total_samples, DESCRIPTOR_DEPTH)
    expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(DESCRIPTORS_ROOT, f"descriptors_anotation_{dataset_for_inference}.bin"), shared = False, nbytes= (total_samples * image_size) * nbytes_float32)).view(total_samples * image_size)

    # saves values for graph scale
    min_value = float(torch.min(expected_value[:]))
    max_value = float(torch.max(expected_value[:]))

    return min_value, max_value

def smooth_histogram(histogram: np.array, by: str = 'mean', filter_size: int = 3):
    # smooths a histogram with the concepts of spatial filtering: to apply a filter by convolving the histogram function (f) with a selected function (g)
    # f * g = Σ f(s) . g(s - t)   # 1D convolution for discrete numbers

    # defines operator for filtering (by)
    filter_operator = {'mean': 1/np.mean(histogram),
                      'median': 1/np.median(histogram)}
    # creates a filter (g function) given filter_size (nx1) and filter_operator
    filter = [filter_operator[by] for i in range(filter_size)]
    # convolves g over f
    smoothed_histogram = scipy.ndimage.convolve1d(histogram, filter)

    return smoothed_histogram

class Statistics():

    def __init__(self, sample_predictions_vector, sample_expected_value):
        self.sample_predictions_vector = torch.tensor(sample_predictions_vector, dtype=torch.float32)
        self.sample_expected_value = torch.tensor(sample_expected_value, dtype=torch.float32)

        self.mean = torch.mean(self.sample_predictions_vector).item()
        self.median = torch.median(self.sample_predictions_vector).item()
        self.mode = torch.mode(self.sample_predictions_vector.flatten())[0].item()#scipy.stats.mode(np.array(sample_predictions_vector).flatten())[0]#torch.linalg.vector_norm(torch.flatten(self.sample_predictions_vector), ord = 5).item()
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
    dataset = load_dataset(dataset_for_inference)
    len_mode = int(len(os.listdir(os.path.join(SAMPLES_PATH, dataset_for_inference)))/3) #TODO alterar isso para abrir a partir do json de metadados
    save_histogram_path = os.path.join(EVALUATION_ROOT, MODEL_VERSION, dataset_for_inference, "histogram")
    save_error_from_cnn1_path = os.path.join(EVALUATION_ROOT, MODEL_VERSION, dataset_for_inference, "error_from_image", "from_cnn1_output")
    save_error_from_image_path = os.path.join(EVALUATION_ROOT, MODEL_VERSION, dataset_for_inference, "error_from_image","from_original_image")
    original_image_path = os.path.join(ORIGINAL_IMAGE_ROOT)

    ### Loads model ###
    model = MODEL_NETWORK#.to('cuda')
      #loss_fn = LOSS_FUNCTION#torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

    checkpoint = torch.load(CHECKPOINT_PATH)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)  # NOTE: Some checkpoints state dicts might not have the expected keys, as seen in  https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    except:
        load_state_dict(model, checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_states'][0])

    ### Training time ###
    # print("Training time")
    # for actual_epoch in tqdm(range(EPOCHS)):
    #     train_loss = train_epoch(model=model, train_loader=dataset, optimizer=optimizer, loss_fn=loss_fn)

    #     print(f"Epoch {actual_epoch + 1}, Loss: {train_loss}")

    # gets the model used
    #model = base_model.model.to('cuda')

    ### Evaluation time ###
    print("Evaluation time")
    partial_loss, predicted_value, expected_value = evaluate(model=model, eval_loader=dataset)

    #saves predicted values for analysis
    #transforms data before saving
    values_ziped = zip(predicted_value, expected_value)  #zips predicted and expected values
    column_array_values = np.array(list(values_ziped))  # converts to numpy
    #saves prediction`s data
    with open(os.path.join(EVALUATION_ROOT, MODEL_VERSION, dataset_for_inference, "predicted_values",  f"{MODEL_VERSION}.txt"), "w") as file: # overrides if file exists
        file.write("predicted_value,expected_value\n")

    with open(os.path.join(EVALUATION_ROOT, MODEL_VERSION, dataset_for_inference, "predicted_values",  f"{MODEL_VERSION}.txt"), "a+") as file:
        for line in column_array_values:
            file.write(f"{line[0]}, {line[1]}\n")

    ### Histograms ###
    print("calculating histograms of predictions\n")
    predicted_value_for_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(predicted_value,( -1, CNN1_OUTPUT_SHAPE, CNN1_OUTPUT_SHAPE)))}
    expected_value_from_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(expected_value,( -1, CNN1_OUTPUT_SHAPE, CNN1_OUTPUT_SHAPE)))}

    min_value, max_value = get_min_max_values(dataset_for_inference)

    for i in range(len_mode):
        values = np.array(predicted_value_for_samples[f'sample_{i}']).flatten() #flattens for histogram calculation
        # calculates statistics
        stats = Statistics(predicted_value_for_samples[f'sample_{i}'], expected_value_from_samples[f'sample_{i}'])

        # define number of bins
        bins = int(math.ceil(max_value/(EXPECTED_RANGE[ANALYTE][0]*0.1/2))) #valor maximo do analito / (metade do pior erro relativo * (10% do menor valor esperado))

        # calculate histogram
        histogram, bins_ranges = np.histogram(a = values, bins = bins, range = (min_value, max_value))

        # smooths histogram with a filter (by) with size n x 1
        corrected_histogram = histogram #smooth_histogram(histogram = histogram, by = 'mean', filter_size = 3)

        # generates matplotlib figure and add a histogram
        plt.figure(figsize=(15, 8))
        #plt.hist(values = histogram, edges = bins_ranges, range = (min_value, max_value), color='black')
        plt.stairs(values = corrected_histogram, edges = bins_ranges, fill=True, color='black')
        #plt.bar(x = bins_ranges[0:-1], height  = histogram , color='black')
        # adds vertical lines for basic statistic values
        plt.axvline(x = stats.mean, alpha = 0.5, c = 'red')
        plt.axvline(x = stats.median, alpha = 0.5, c = 'blue')
        plt.axvline(x = stats.mode, alpha = 0.5, c = 'green')
        plt.axvline(x = expected_value_from_samples[f'sample_{i}'][0][0], alpha = 0.5, c = 'grey')

        plt.title(f"Sample_{i}, Expected Value: {round(expected_value_from_samples[f'sample_{i}'][0][0], 2)}")
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

        ##text settings##
        # expected value
        plt.text(x = x_max+0.5,  y=y_max - 9.92 * scale_factor,
                s = f" value:")
        plt.text(x = x_max + text, y=y_max - 9.92 * scale_factor,
                s = f" {expected_value_from_samples[f'sample_{i}'][0][0]:.2f}", c = 'grey', alpha = 0.6)
        # mean
        plt.text(x = x_max+0.5,  y=y_max - 10.20 * scale_factor,
                s = f" mean:")
        plt.text(x = x_max + text, y=y_max - 10.20 * scale_factor,
                s = f" {stats.mean:.2f}", c = 'red', alpha = 0.6)
        # median
        plt.text(x = x_max ,  y=y_max - 10.49 * scale_factor,
                s = f" median:" )
        plt.text(x = x_max + text_median, y=y_max - 10.49 * scale_factor,
                s = f"  {stats.median:.2f}", c = 'blue', alpha = 0.6)
        # mode
        plt.text(x = x_max ,  y=y_max - 10.78 * scale_factor,
                s = f" mode:" )
        plt.text(x = x_max + text,  y=y_max - 10.78 * scale_factor,
                s = f" {stats.mode:.2f}", c = 'green', alpha = 0.6)
        # stats
        plt.text(x = x_max ,  y=y_max - 12.4 * scale_factor,
                s = f" var: {stats.variance:.2f}\n std: {stats.std:.2f}\n mad: {stats.mad:.2f}\n min: {stats.min_value}\n max: {stats.max_value:.2f}", c = 'black')

        #plt.text(x = max_value + 0.5,  y = 0,
        #         s = f" mean: {stats.mean:.2f}\n median: {stats.median:.2f}\n mode: {stats.mode:.2f}\n var: {stats.variance:.2f}\n std: {stats.std:.2f}\n min: {stats.min_value:.2f}\n max: {stats.max_value:.2f}")
        plt.savefig(os.path.join(save_histogram_path, f"sample_{i}.png"))
        plt.close('all')

    # reshapes to the size of the output from the first cnn in vgg11  and the total of images
    print("reshaping images to match cnn1 output\n")
    partial_loss_from_cnn1_output = np.reshape(partial_loss, (-1, CNN1_OUTPUT_SHAPE, CNN1_OUTPUT_SHAPE))

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
    main(dataset_for_inference=IMAGES_TO_EVALUATE)