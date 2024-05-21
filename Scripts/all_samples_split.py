import os
import random
import shutil

from tqdm import tqdm

ANALYTE = "Chloride"
DATASET_SPLIT = 0.8

PATH = [f"F:\\Mestrado\\{ANALYTE}_Samples"]
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

def split_folders(folders, split_ratio = DATASET_SPLIT , seed=42):
    random.seed(seed)
    random.shuffle(folders)
    split_index = int(len(folders) * split_ratio)
    return folders[:split_index], folders[split_index:]

def copy_files(folders, save_path):
    for folder in tqdm(folders, desc="Copying files"):
        destination_folder = os.path.join(save_path, os.path.basename(folder))
        os.makedirs(destination_folder, exist_ok=True)
        for filename in os.listdir(folder):
            src_file = os.path.join(folder, filename)
            dst_file = os.path.join(destination_folder, filename)
            shutil.copy(src_file, dst_file)


def main():
    samples_paths = all_dirs(PATH)
    train_split, test_split = split_folders(samples_paths)

    copy_files(train_split, os.path.join(SAVE_PATH, "train_samples", f"{ANALYTE}"))
    copy_files(test_split, os.path.join(SAVE_PATH, "test_samples", f"{ANALYTE}"))

if __name__ == "__main__":
    main()
print(' ')