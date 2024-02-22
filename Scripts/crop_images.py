import chemical_analysis as ca
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import os

from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset

os.makedirs("../images", exist_ok =True)
os.makedirs("../cache_dir", exist_ok =True)

#preprocessamento dos dados 
samples = AlkalinitySampleDataset(
    base_dirs = '../Alkalinity_Samples',  
    progress_bar = True, 
    skip_blank_samples = True, 
    skip_incomplete_samples = True, 
    skip_inference_sample= True, 
    skip_training_sample = False, 
    verbose = True
)

processed_samples = ProcessedAlkalinitySampleDataset(
    dataset = samples, 
    cache_dir = '../cache_dir',
    num_augmented_samples = 0, 
    progress_bar = True, 
    transform = None,
)


#crop centralizado

for i, _ in enumerate(processed_samples):
    print(f"Imagem {i}")
    try:
        #gets the mask for that sample
        mask =  processed_samples[i].sample_analyte_mask

        nonzero_rows, nonzero_cols = np.nonzero(mask)
        min_row, max_row = min(nonzero_rows), max(nonzero_rows)
        min_col, max_col = min(nonzero_cols), max(nonzero_cols)

        #cropp based on mask 
        actual_image = processed_samples[i].sample_bgr_image[min_row:max_row, min_col:max_col]

        image_heigth, image_width = actual_image.shape[0], actual_image.shape[1]

        #cropp for vgg input
        cropped_image = actual_image[int(image_heigth/2)-112:int(image_heigth/2)+112, int(image_width/2)-112:int(image_width/2)+112]

        #saves images
        plt.imsave(f"../images/sample_{i}.png", cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)/255)

        #saves alkalinity value
        with open(f"../images/sample_{i}.txt", "w", encoding='utf-8') as f:
            json.dump(processed_samples.alkalinity_values[i]['theoreticalValue'],f, ensure_ascii=False, indent=4)
    except: 
        print(f"Imagem problematica : {i}")



# #salva as imagens
# for i, image in enumerate(cropped_images):
#     plt.imsave(f"./images/sample_{i}.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255)
#     with open(f"./images/sample_{i}.txt", "w", encoding='utf-8') as f:
#         json.dump(processed_samples.alkalinity_values[i]['theoreticalValue'],f, ensure_ascii=False, indent=4)