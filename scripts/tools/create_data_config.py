import argparse
import os
from pathlib import Path
from typing import List


def create_data_prefix(list_of_paths: List[str]):
    list_of_bin_files = []
    # Select all .bin files
    for path in list_of_paths:
        path_to_files = [os.path.join(dp, f) for dp, _, fn in os.walk(os.path.expanduser(path)) for f in fn]
        list_of_bin_files.extend(
            [raw_file for raw_file in path_to_files if Path(raw_file).suffix.lower().endswith(".bin")]
        )

    list_of_sizes = [os.path.getsize(path) for path in list_of_bin_files]
    total_tokens = sum(list_of_sizes)
    list_of_normalized_tokens = [float(i) / total_tokens for i in list_of_sizes]

    list_of_bin_files = [
        bin_file[:-4] for bin_file in list_of_bin_files
    ]  # NOTE(tj.solergibert) Delete .bin extension to have file prefixes
    interleaved = [val for pair in zip(list_of_normalized_tokens, list_of_bin_files) for val in pair]
    print(*interleaved, sep=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        required=True,
        help="Comma separated list of paths to generate the config from. e.g. -p /path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C",
    )
    args = parser.parse_args()

    paths = [x.strip() for x in args.paths.split(",")]
    create_data_prefix(paths)
