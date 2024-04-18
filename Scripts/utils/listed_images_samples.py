#disabled , v2 is better

# import os
# import json
# import shutil

# from chemical_analysis import AlkalinitySampleDataset, ProcessedAlkalinitySampleDataset
# from tqdm import tqdm
# from typing import List, Any, Dict, Tuple

# ANALYTE = "Alkalinity"
# os.makedirs(os.path.join(os.path.dirname(__file__),"..", "..", "Train_Test_Samples", f"{ANALYTE}_Samples"), exist_ok =True)

# SAVE_PATH = os.path.join(os.path.dirname(__file__),"..", "..", "Train_Test_Samples", f"{ANALYTE}_Samples")

# SAMPLES_BASE_PATH = [r"F:\\Mestrado\\Alkalinity_Samples"]

# computing = {
#             "Alkalinity": {"dataset": AlkalinitySampleDataset,
#                  "processed_dataset": ProcessedAlkalinitySampleDataset}
# }

# dataset_process = computing(ANALYTE.get("dataset"))
# dataset_processed = computing(ANALYTE.get("processed_dataset"))

# i = 0
# for image in jpegs.values():
#     try:
#         new_name = os.rename(a, f"sample_{i}")
#         shutil.copy(image, SAVE_PATH )
#     except:
#         print(f"problem with image: {image}")
#         break

# i = 0
# for json in jsons.values()[1]:
#     try:
#         new_name = os.rename(a, f"sample_{i}")
#         shutil.copy(json, SAVE_PATH)
#     except:
#         print(f"problem with json: {json}")
#         break


# print("Done!")