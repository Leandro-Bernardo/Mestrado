import os
import json
import shutil

from tqdm import tqdm
from typing import  Tuple, List, Dict, Any

#variables
ANALYTE = "Alkalinity"
PATH = [f"..\\..\\{ANALYTE}_Samples"]#os.path.join(os.path.dirname(__file__), "..", f"{ANALYTE}_Samples")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "blank_files", f"{ANALYTE}")

os.makedirs(SAVE_PATH, exist_ok = True)

def all_files_and_dirs(path: List[str]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    jpegs: Dict[str, str] = list()  # format: [jpegs, ...]
    dirs: List[str] = list()         # format: [dir, ...]
    blanks: List[str] = list()       # format: [blanks, ...]
    extra_files: List[str] = list()  # format: [extra_file_1, extra_file_2, ...]

    not_visited_dirs = list(path)

    with tqdm(desc="Scanning folders", total=len(path)) as pbar:
        while len(not_visited_dirs) != 0:
            current_dir = not_visited_dirs.pop(0)
            dirs.append(current_dir)

            for filename in sorted(os.listdir(current_dir)):
                path = os.path.join(current_dir, filename)
                if os.path.isfile(path):
                    if filename.lower().endswith(".json"):
                        with open(path, "r", encoding="utf8") as file:
                            data = json.load(file)
                            blank_file = data["sample"]["blankFileName"]
                            jpeg_file = data["sample"]["fileName"]
                            extra_jpegs = data["sample"]["extraFileNames"]

                            if blank_file is None:  # takes only the blank files (which does not refer to any blank file. Hence, its taged with None in this field)
                                blanks.append(os.path.join(current_dir, filename))
                                jpegs.append(os.path.join(current_dir, jpeg_file))
                                for extra_jpeg in extra_jpegs:
                                    extra_files.append(os.path.join(current_dir, extra_jpeg))

                    elif filename.lower().endswith(".jpg"):
                        pass

                elif os.path.isdir(path):
                    not_visited_dirs.append(path)
                    pbar.total += 1
                    pbar.refresh()

            pbar.update(1)
    return blanks, jpegs, extra_files

blanks_path, jpeg_path, extra_files_path = all_files_and_dirs(PATH)



for blank_file in blanks_path:
    try:
        shutil.copy(blank_file, SAVE_PATH)
    except:
        print(f"{json_file} operation failed")

for jpeg_file in jpeg_path:
    try:
        shutil.copy(jpeg_file, SAVE_PATH)
    except:
        print(f"{jpeg_file} operation failed")

for extra_file in extra_files_path:
    try:
        shutil.copy(extra_file, SAVE_PATH)
    except:
        print(f"{extra_file} operation failed")



