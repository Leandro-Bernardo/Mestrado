# disabled, implemented as feature at crop_images.py

# import chemical_analysis as ca
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import json
# import os

# from chemical_analysis.alkalinity import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset

# #variables
# ANALYTE = "Alkalinity"
# SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Alkalinity_Samples")
# CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cache_dir")
# SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "images")
# TRAIN_TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Train_Test_Samples", f"{ANALYTE}_Samples")

# #defines path dir
# os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..","cache_dir"), exist_ok =True)

# #data preprocessing
# samples = AlkalinitySampleDataset(
#     base_dirs = SAMPLES_PATH,
#     progress_bar = True,
#     skip_blank_samples = True,
#     skip_incomplete_samples = True,
#     skip_inference_sample= True,
#     skip_training_sample = False,
#     verbose = True
# )

# processed_samples = ProcessedAlkalinitySampleDataset(
#     dataset = samples,
#     cache_dir = CACHE_PATH,
#     num_augmented_samples = 0,
#     progress_bar = True,
#     transform = None,
# )


# #save images for train test of pmf based model
# for i, _ in enumerate(processed_samples):
#     plt.imsave(f"{TRAIN_TEST_PATH}/sample_{i}.png", cv2.cvtColor(processed_samples[i].sample_bgr_image, cv2.COLOR_BGR2RGB)/255)
# with open(f"{TRAIN_TEST_PATH}/sample_{i}.txt", "w", encoding='utf-8') as f:
#     json.dump(processed_samples.alkalinity_values[i]['theoreticalValue'],f, ensure_ascii=False, indent=4)
