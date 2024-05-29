import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math

from tqdm import tqdm
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import FloatTensor, UntypedStorage

if not torch.cuda.is_available():
    assert("cuda isnt available")

else:
    device = "cuda"

# variables
ANALYTE = "Chloride"
SKIP_BLANK = False

if ANALYTE == "Alkalinity":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    RECEPTIVE_FIELD_DIM = 15
    IMAGE_SHAPE = 97
    IMAGE_SIZE = IMAGE_SHAPE * IMAGE_SHAPE # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    DESCRIPTOR_DEPTH = 448

elif ANALYTE == "Chloride":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_3'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = False
    RECEPTIVE_FIELD_DIM = 27
    IMAGE_SHAPE = 86
    IMAGE_SIZE = IMAGE_SHAPE * IMAGE_SHAPE  # after the crop based on the receptive field  (shape = (112 - 27, 112 - 27))
    DESCRIPTOR_DEPTH = 1472

if ANALYTE == "Phosphate":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    #RECEPTIVE_FIELD_DIM = 15
    #IMAGE_SIZE = 97 * 97  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    #DESCRIPTOR_DEPTH = 448

if ANALYTE == "Sulfate":
    EPOCHS = 1
    LR = 0.001
    BATCH_SIZE = 64
    EVALUATION_BATCH_SIZE = 1
    GRADIENT_CLIPPING_VALUE = 0.5
    CHECKPOINT_SAVE_INTERVAL = 25
    MODEL_VERSION = 'model_1'
    DATASET_SPLIT = 0.8
    USE_CHECKPOINT = True
    #RECEPTIVE_FIELD_DIM = 15
    #IMAGE_SIZE = 97 * 97  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))
    #DESCRIPTOR_DEPTH = 448

if SKIP_BLANK:
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank"))
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "Udescriptors", "no_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}",  "no_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "Udescriptors","no_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank")
else:
    SAMPLES_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank"))
    IDENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "images",f"{ANALYTE}", "with_blank")
    CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints", f"{ANALYTE}", "Udescriptors", "with_blank")
    DESCRIPTORS_ROOT = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank")
    EVALUATION_ROOT = os.path.join(os.path.dirname(__file__), "evaluation", f"{ANALYTE}", "Udescriptors", "with_blank")
    ORIGINAL_IMAGE_ROOT = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank")
'F:\\Mestrado\\Scripts\\evaluation\\Chloride\\Udescriptors\\with_blank\\histogram\\model_3\\test\\sample_0.png'
LAST_CHECKPOINT = sorted(os.listdir(os.path.join(CHECKPOINT_ROOT, MODEL_VERSION)), key = lambda x: int(x.split('_')[-1]))[-1]
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, MODEL_VERSION, LAST_CHECKPOINT)

EXPECTED_RANGE = {
                "Alkalinity": (500.0, 2500.0),
                "Chloride": (10000.0, 300000.0),
                "Phosphate": (0.0, 50.0),
                "Sulfate":(0.0, 4000.0),
                 }

print('Using this checkpoint:', CHECKPOINT_PATH)

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


# model definition
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

checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# utilities functions and classes

# loads data
def load_files(starting_point, mode):

    if mode == 'train':
        batch_size = BATCH_SIZE
        sample_path = os.path.join(SAMPLES_PATH, "train")
        descriptors_path = os.path.join(DESCRIPTORS_ROOT, "train")
        shuffle = True

    elif mode == 'test':
        batch_size = EVALUATION_BATCH_SIZE
        sample_path = os.path.join(SAMPLES_PATH, "test")
        descriptors_path = os.path.join(DESCRIPTORS_ROOT, "test")
        shuffle = False

    else:
        raise NotImplementedError

    total_samples = int(len(os.listdir(sample_path)) / 3)
    dim = total_samples * IMAGE_SIZE

    # reads the untyped storage object of saved descriptors
    descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(descriptors_path, "descriptors.bin"), shared=False, nbytes=(dim * DESCRIPTOR_DEPTH) * torch.finfo(torch.float32).bits // 8)).view(IMAGE_SIZE * total_samples, DESCRIPTOR_DEPTH)
    expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(descriptors_path, "descriptors_anotation.bin"), shared=False, nbytes=(dim) * torch.finfo(torch.float32).bits // 8)).view(IMAGE_SIZE * total_samples)

    X_values = descriptors[starting_point: starting_point + batch_size, :].to('cuda')
    y_values = expected_value[starting_point: starting_point + batch_size].to('cuda')

    dataset = DataLoader(list(zip(X_values, y_values)), batch_size = batch_size, shuffle= shuffle)
    return dataset

def train_epoch(model, train_loader, optimizer, loss_fn):
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

def evaluate(model, eval_loader, loss_fn):

    model.eval()  # change model to evaluation mode

    total_samples = len(eval_loader)

    with torch.no_grad():
        for X_batch, y_batch in eval_loader:

            y_pred = model(X_batch).squeeze(1)
            predicted_value = round(y_pred.item(), 2)
            expected_value = y_batch.item()

            loss = loss_fn(y_pred, y_batch)
            partial_loss = loss.item()

            #_, predicted = torch.max(y_pred, 1)
            #correct_predictions += (predicted == y_batch).sum().item()

   # accuracy = correct_predictions / total_samples

    return partial_loss, predicted_value, expected_value # ,accuracy

def get_min_max_values(mode):

    if mode == 'train':
        sample_path = os.path.join(SAMPLES_PATH, "train")
        descriptors_path = os.path.join(DESCRIPTORS_ROOT, "train")

    elif mode == 'test':
        sample_path = os.path.join(SAMPLES_PATH, "test")
        descriptors_path = os.path.join(DESCRIPTORS_ROOT, "test")

    else:
        raise NotImplementedError

    total_samples = int(len(os.listdir(sample_path)) / 3)
    dim = total_samples * IMAGE_SIZE

    # reads the untyped storage object of saved descriptors
    #descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(descriptors_path, "descriptors.bin"), shared=False, nbytes=(dim * DESCRIPTOR_DEPTH) * torch.finfo(torch.float32).bits // 8)).view(IMAGE_SIZE * total_samples, DESCRIPTOR_DEPTH)
    expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(descriptors_path, "descriptors_anotation.bin"), shared=False, nbytes=(dim) * torch.finfo(torch.float32).bits // 8)).view(IMAGE_SIZE * total_samples)

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

def main(sample_to_evaluate):
    train_samples_len = (int(len(os.listdir(os.path.join(SAMPLES_PATH, "train")))/3))
    test_samples_len = (int(len(os.listdir(os.path.join(SAMPLES_PATH, "test")))/3))

    if sample_to_evaluate == "train":
        len_mode = train_samples_len

        save_histogram_path = os.path.join(EVALUATION_ROOT, "train", "histogram", MODEL_VERSION)
        save_error_from_cnn1_path = os.path.join(EVALUATION_ROOT, "train", "error_from_image", "from_cnn1_output", MODEL_VERSION)
        save_error_from_image_path = os.path.join(EVALUATION_ROOT, "train", "error_from_image","from_original_image", MODEL_VERSION)
        original_image_path = os.path.join(ORIGINAL_IMAGE_ROOT, "train")

    elif sample_to_evaluate == "test":
         len_mode = test_samples_len
         save_histogram_path = os.path.join(EVALUATION_ROOT, "test", "histogram", MODEL_VERSION, )
         save_error_from_cnn1_path = os.path.join(EVALUATION_ROOT, "test", "error_from_image", "from_cnn1_output", MODEL_VERSION)
         save_error_from_image_path = os.path.join(EVALUATION_ROOT,  "test", "error_from_image", "from_original_image", MODEL_VERSION)
         original_image_path = os.path.join(ORIGINAL_IMAGE_ROOT, "test")

    else:
        NotImplementedError

    #training
    print("Training time")
    for actual_epoch in tqdm(range(EPOCHS)):
        starting_point = 0
        batches_loss = 0
        train_samples_len = (int(len(os.listdir(os.path.join(SAMPLES_PATH, "train")))/3))
        num_batches = int(train_samples_len * IMAGE_SIZE / BATCH_SIZE)

        for _ in range(num_batches):
                sample_batch = load_files(starting_point, mode = "train")
                loss = train_epoch(model, sample_batch, optimizer, loss_fn)
                batches_loss += loss
                starting_point += BATCH_SIZE

    mean_loss = batches_loss/num_batches

    print(f"Epoch {actual_epoch + 1}, Loss: {mean_loss}")

    #evaluation
    print("Evaluation time")
    starting_point = 0
    partial_loss = []
    predicted_value = []
    expected_value = []

    num_batches = int(len_mode * IMAGE_SIZE / EVALUATION_BATCH_SIZE)

    for _ in range(num_batches):
        sample_batch = load_files(starting_point, mode = sample_to_evaluate)
        this_partial_loss, this_predicted_value, this_expected_value = evaluate(model, sample_batch, loss_fn)

        partial_loss.append(this_partial_loss), predicted_value.append(this_predicted_value), expected_value.append(this_expected_value)

        starting_point+= EVALUATION_BATCH_SIZE

    partial_loss = np.array(partial_loss)
    predicted_value = np.array(predicted_value)
    expected_value = np.array(expected_value)

    with open(os.path.join(EVALUATION_ROOT, sample_to_evaluate, "predicted_values", MODEL_VERSION, f"{MODEL_VERSION}.txt"), "w") as file: # overrides if file exists
        file.write("predicted_value,expected_value\n")

    with open(os.path.join(EVALUATION_ROOT, sample_to_evaluate, "predicted_values", MODEL_VERSION, f"{MODEL_VERSION}.txt"), "a+") as file:
        for i in range(len(predicted_value)):
            file.write(f"{predicted_value[i]},{expected_value[i]}\n")

    #histograms
    print("calculating histograms of predictions\n")
    predicted_value_for_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(predicted_value,( -1, IMAGE_SHAPE, IMAGE_SHAPE)))}
    expected_value_from_samples = {f"sample_{i}" : value for i, value in enumerate(np.reshape(expected_value,( -1, IMAGE_SHAPE, IMAGE_SHAPE)))}

    min_value, max_value = get_min_max_values(sample_to_evaluate)

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
    #main(sample_to_evaluate="train")
    main(sample_to_evaluate="test")