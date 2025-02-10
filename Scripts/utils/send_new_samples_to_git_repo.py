from argparse import Namespace
from git import RemoteProgress
from git.repo import Repo
from tqdm import tqdm
from typing import List
import argparse, os


# Set default values for command line arguments.
DEFAULT_PATH = "../../Alkalinity_Samples"
DEFAULT_MAX_PUSH_SIZE_IN_GB = 1.5  # Maximum of 2.0 in GitHub.
DEFAULT_COMMIT_MESSAGE = "Inclusão de amostras via script"

print(DEFAULT_PATH)
# Progress bar for push.
class PushProgress(RemoteProgress):
    def __init__(self, pbar: tqdm):
        super().__init__()
        self.pbar = pbar

    def update(self, op_code, cur_count, max_count=None, message=""):
        self.pbar.set_description(f"Pushing ({message})" if message != "" else "Pushing")
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


# Create groups of files with a maximum max_push_size_in_bytes each.
def make_groups(root: str, relative_filepaths: List[str], max_push_size_in_bytes: float) -> List[List[str]]:
    groups = list()
    current_group = list()
    current_group_size = 0
    for relative_filepath in tqdm(sorted(relative_filepaths), total=len(relative_filepaths), leave=False, desc="Grouping untracked files"):
        size = os.path.getsize(os.path.join(root, relative_filepath))
        if current_group_size + size < max_push_size_in_bytes:
            current_group.append(relative_filepath)
            current_group_size += size
        else:
            groups.append(current_group)
            current_group = [relative_filepath]
            current_group_size = size
    if len(current_group) != 0:
        groups.append(current_group)
    return groups


# Declare the mais procedure.
def main(args: Namespace) -> None:
    # Create a repo object to have high-level access to the repository.
    repo = Repo(args.path)
    # Get the list of untracked files and create groups with a maximum of args.max_push_size_in_gb Gb each.
    groups = make_groups(args.path, repo.untracked_files, args.max_push_size_in_gb * 1073741824.0)
    # For each group...
    for group in tqdm(groups, leave=False, desc="Pushing groups of files"):
        # Add untracked files.
        for filepath in tqdm(group, leave=False, desc="Adding files"):
            repo.git.add(filepath)
        # Commit and push changes.
        repo.index.commit(args.commit_message)
        origin = repo.remote("origin")
        with tqdm(leave=False) as pbar:
            origin.push(progress=PushProgress(pbar))
    # That's it!
    print("Done!")


# Make this script behave like a executable.
if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", metavar="PATH", type=str, default=DEFAULT_PATH)
    parser.add_argument("--max_push_size_in_gb", metavar="SIZE", type=float, default=DEFAULT_MAX_PUSH_SIZE_IN_GB)
    parser.add_argument("--commit_message", metavar="MESSAGE", type=str, default=DEFAULT_COMMIT_MESSAGE)
    args = parser.parse_args()
    # Call the main procedure.
    main(args)
