import os
import random
import shutil

from tqdm import tqdm

ANALYTE = "Alkalinity"
DATASET_TRAIN_TEST_SPLIT = 0.8
DATASET_TRAIN_VAL_SPLIT = 0.8

PATH = [f"Y:\\Leandro-Bernardo\\Mestrado\\{ANALYTE}_Samples"]
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..")
def all_dirs(base_dirs):
    dirs = list()        # [dir, ...]
    not_visited_dirs = list(base_dirs)
    samples_dirs = list()
    with tqdm(desc="Scanning folders", total=len(base_dirs)) as pbar:
        while len(not_visited_dirs) != 0:
            current_dir = not_visited_dirs.pop(0)
            dirs.append(current_dir)
            for filename in sorted(os.listdir(current_dir)):
                path = os.path.join(current_dir, filename)
                if os.path.isfile(path):
                    if filename.lower().endswith(".json") or filename.lower().endswith(".jpg"):
                        if current_dir not in samples_dirs:
                            samples_dirs.append(current_dir)
                elif os.path.isdir(path):
                    not_visited_dirs.append(path)
                    pbar.total += 1
                    pbar.refresh()
            pbar.update(1)
    return samples_dirs

def split_folders(folders, split_train_test_ratio = DATASET_TRAIN_TEST_SPLIT, split_train_val_ratio = DATASET_TRAIN_VAL_SPLIT, seed=42):
    random.seed(seed)

    # splits train and test
    random.shuffle(folders)
    split_index = int(len(folders) * split_train_test_ratio)
    train_folders, test_folders = folders[:split_index], folders[split_index:]

    # splits train and validation
    random.shuffle(train_folders)
    split_index = int(len(train_folders) * split_train_val_ratio)
    train_folders, val_folders = train_folders[:split_index], train_folders[split_index:]
    return train_folders, val_folders, test_folders

def copy_files(folders, stage, save_path):
    for folder in tqdm(folders, desc=f"Copying {stage} files"):
        destination_folder = os.path.join(save_path, os.path.basename(folder))
        os.makedirs(destination_folder, exist_ok=True)
        for filename in os.listdir(folder):
            src_file = os.path.join(folder, filename)
            dst_file = os.path.join(destination_folder, filename)
            shutil.copy(src_file, dst_file)


def main():
    samples_paths = all_dirs(PATH)
    train_folders, val_folders, test_folders = split_folders(samples_paths)

    copy_files(train_folders, "train", os.path.join(SAVE_PATH, "train_samples", f"{ANALYTE}"))
    copy_files(val_folders, "val", os.path.join(SAVE_PATH, "val_samples", f"{ANALYTE}"))
    copy_files(test_folders, "test", os.path.join(SAVE_PATH, "test_samples", f"{ANALYTE}"))

if __name__ == "__main__":
    main()
