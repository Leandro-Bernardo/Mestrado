import chemical_analysis as ca
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json, yaml
import os

from typing import Tuple, List, Dict, Any
from tqdm import tqdm
from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset
from chemical_analysis.chloride import ChlorideSampleDataset, ProcessedChlorideSampleDataset
from chemical_analysis.ph import PhSampleDataset, ProcessedPhSampleDataset
#from chemical_analysis.sulfate import SulfateSampleDataset, ProcessedSulfateSampleDataset
#from chemical_analysis.phosphate import PhosphateSampleDataset, ProcessedPhosphateSampleDataset

### Variables ###
# reads setting`s json
with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    ANALYTE = settings["analyte"]
    SKIP_BLANK = settings["skip_blank"]
    PROCESS_BLANK_FILES_SEPARATEDLY = settings["process_blank_files_separatedly"]
    IMAGE_SIZE = settings["image_size"]

SAMPLES_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "Splited_Samples",  f"{ANALYTE}",  "train_samples")
SAMPLES_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "Splited_Samples",  f"{ANALYTE}",  "val_samples")
SAMPLES_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "Splited_Samples",  f"{ANALYTE}",   "test_samples")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache_dir")

if SKIP_BLANK == True and  PROCESS_BLANK_FILES_SEPARATEDLY == False:  # dont use blanks
    SAVE_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", "train")
    SAVE_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", "val")
    SAVE_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "no_blank", "test")

elif SKIP_BLANK == False and PROCESS_BLANK_FILES_SEPARATEDLY == False:  # use blanks and process it together
    SAVE_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", "train")
    SAVE_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", "val")
    SAVE_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "with_blank", "test")

elif SKIP_BLANK == False and PROCESS_BLANK_FILES_SEPARATEDLY == True:  # process blanks separatedly
    SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "blank_files", f"{ANALYTE}")
    SAVE_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank", "train")
    SAVE_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank", "val")
    SAVE_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "images", f"{ANALYTE}", "processed_blank", "test")

elif SKIP_BLANK == True and PROCESS_BLANK_FILES_SEPARATEDLY == True:  # missmatch combination
    raise Exception('''
    Missmatch combinations
        Case: SKIP_BLANK == True and PROCESS_BLANK_FILES_SEPARATEDLY == True)

    OPTIONS:
        SKIP_BLANK must be  True   and  PROCESS_BLANK_FILES_SEPARATEDLY  False  for not to process blanks, or
        SKIP_BLANK must be  False  and  PROCESS_BLANK_FILES_SEPARATEDLY  False  for use blanks and process it together, or
        SKIP_BLANK must be  False  and  PROCESS_BLANK_FILES_SEPARATEDLY  True   for process blanks separatedly
        ''')

#makes path dir
os.makedirs(SAVE_TRAIN_PATH, exist_ok = True)
os.makedirs(SAVE_VAL_PATH, exist_ok = True)
os.makedirs(SAVE_TEST_PATH, exist_ok = True)
os.makedirs(CACHE_PATH, exist_ok = True)


def main(sample_path: str, save_path: str, stage:str):
    dataset_processor = {"Alkalinity":{"dataset": AlkalinitySampleDataset, "processed_dataset": ProcessedAlkalinitySampleDataset},
                        "Chloride": {"dataset": ChlorideSampleDataset, "processed_dataset": ProcessedChlorideSampleDataset},
                        #"Sulfate": {"dataset": SulfateSampleDataset, "processed_dataset": ProcessedSulfateSampleDataset},
                        #"Phosphate": {"dataset": PhosphateSampleDataset, "processed_dataset": ProcessedPhosphateSampleDataset},
                        "Ph": {"dataset": PhSampleDataset, "processed_dataset": ProcessedPhSampleDataset},
                        }

    pca_stats = {
                #"Alkalinity": {"lab_mean": np.load(ca.alkalinity.PCA_STATS)['lab_mean'], "lab_sorted_eigenvectors": np.load(ca.alkalinity.PCA_STATS)['lab_sorted_eigenvectors']},
                "Chloride"  : {"lab_mean": np.load(ca.chloride.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.chloride.PCA_STATS)['lab_sorted_eigenvectors']},
                #"Sulfate"   : {"lab_mean": np.load(ca.sulfate.PCA_STATS)['lab_mean']   , "lab_sorted_eigenvectors": np.load(ca.sulfate.PCA_STATS)['lab_sorted_eigenvectors']},
                #"Phosphate" : {"lab_mean": np.load(ca.phosphate.PCA_STATS)['lab_mean'] , "lab_sorted_eigenvectors": np.load(ca.phosphate.PCA_STATS)['lab_sorted_eigenvectors']}
                }

    SampleDataset = dataset_processor[f"{ANALYTE}"]["dataset"]
    ProcessedSampleDataset = dataset_processor[f"{ANALYTE}"]["processed_dataset"]

    #data preprocessing
    samples = SampleDataset(
        base_dirs = sample_path,
        progress_bar = True,
        skip_blank_samples = SKIP_BLANK,
        skip_incomplete_samples = True,
        skip_inference_sample= True,
        skip_training_sample = False,
        verbose = True
    )

    if ANALYTE == "Alkalinity":
        processed_samples = ProcessedSampleDataset(
        dataset = samples,
        cache_dir = CACHE_PATH,
        num_augmented_samples = 0,
        progress_bar = True,
        transform = None, )

    elif ANALYTE == "Chloride":
        processed_samples = ProcessedSampleDataset(
            dataset = samples,
            cache_dir = CACHE_PATH,
            num_augmented_samples = 0,
            progress_bar = True,
            transform = None,
            lab_mean= pca_stats[f"{ANALYTE}"]['lab_mean'],
            lab_sorted_eigenvectors = pca_stats[f"{ANALYTE}"]['lab_sorted_eigenvectors'])

    elif ANALYTE == "Ph":
        processed_samples = ProcessedSampleDataset(
        dataset = samples,
        cache_dir = CACHE_PATH,
        num_augmented_samples = 0,
        progress_bar = True,
        transform = None, )

    #centered crop
    count_of_valid_samples = 0
    for i, _ in enumerate(processed_samples):
        try:
            print(f" Cropping images from {stage} samples {i+1}/{len(processed_samples)}")

            #gets the mask for that sample
            mask =  processed_samples[i].sample_analyte_mask

            nonzero_rows, nonzero_cols = np.nonzero(mask)
            min_row, max_row = min(nonzero_rows), max(nonzero_rows)
            min_col, max_col = min(nonzero_cols), max(nonzero_cols)

            #cropp based on mask
            actual_image = processed_samples[i].sample_bgr_image[min_row:max_row, min_col:max_col]

            image_heigth, image_width = actual_image.shape[0], actual_image.shape[1]

            #cropp for vgg input
            crop = IMAGE_SIZE/2
            cropped_image = actual_image[int(image_heigth/2-crop) : int(image_heigth/2+crop), int(image_width/2-crop) : int(image_width/2+crop)]

            #saves images
            plt.imsave(f"{save_path}/sample_{count_of_valid_samples}.png", cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)/255)

            #saves analyte value
            if ANALYTE != 'Ph':
                with open(f"{save_path}/sample_{count_of_valid_samples}.txt", "w", encoding='utf-8') as f:
                    json.dump(processed_samples.analyte_values[i]['theoreticalValue'], f, ensure_ascii=False, indent=4)
            elif ANALYTE == 'Ph': # takes the raw value from json (and not the theoretical value) (exception because pH doesnt need the theoretical value correction)
                with open(f"{save_path}/sample_{count_of_valid_samples}.txt", "w", encoding='utf-8') as f:
                    json.dump(processed_samples.analyte_values[i]['sourceStock']['value'], f, ensure_ascii=False, indent=4)

            #saves analyte identifier
            with open(f"{save_path}/sample_{count_of_valid_samples}_identity.txt", "w", encoding='utf-8') as f:
                json.dump(processed_samples[i].datetime, f, ensure_ascii=False, indent=4)
                f.write('\n')
                json.dump(processed_samples[i].analyst_name, f, ensure_ascii=False, indent=4)
                f.write('\n')
                json.dump(processed_samples[i].sample_prefix + '.jpg', f, ensure_ascii=False, indent=4)
                f.write('\n')
                try:
                    json.dump(processed_samples[i].blank_prefix + '.jpg', f, ensure_ascii=False, indent=4)
                except:
                    json.dump(processed_samples[i].blank_prefix, f, ensure_ascii=False, indent=4)

            count_of_valid_samples+=1

        except:
            print(f"Imagem problematica : {processed_samples[i].analyst_name} {processed_samples[i].datetime}, sample:  {processed_samples[i].sample_prefix}")


if __name__ == "__main__":
    main(SAMPLES_TRAIN_PATH, SAVE_TRAIN_PATH, "train")
    main(SAMPLES_VAL_PATH, SAVE_VAL_PATH, "validation")
    main(SAMPLES_TEST_PATH, SAVE_TEST_PATH, "test")