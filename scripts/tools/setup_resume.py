import glob
import os
import sys


def main(ckpt_dir):
    # Get most recent checkpoint
    list_of_files = glob.glob(f"{ckpt_dir}/*")
    list_of_folders = [file for file in list_of_files if os.path.isdir(file)]
    latest_folder = max(list_of_folders, key=os.path.getctime)
    # Return correct format
    print(latest_folder.replace("=", "\="))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 setup_resume.py /path/to/ckpt/dir"
    main(sys.argv[1])
