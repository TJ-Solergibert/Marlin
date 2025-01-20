import os
import shutil
import sys
from pathlib import Path
from typing import Union

UNFINISHED_CHECKPOINT_SUFFIX = "-unfinished"  # NOTE(tj.solergibert) NeMo Default https://github.com/NVIDIA/NeMo/blob/d98b7cd1b1a9c06d094a8b349bcc2e7b5a7e16a9/nemo/utils/callbacks/nemo_model_checkpoint.py#L45C5-L45C49


def format_checkpoint_unfinished_marker_path(checkpoint_path: Union[Path, str]) -> Path:
    """Format the path to the unfinished checkpoint marker file.

    If the marker file exists, corresponding checkpoint is considered unfinished/incomplete.
    NOTE: Marker path for the EMA checkpoint part is the same as for the original checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file or dir.
            Does not need to exist.

    Returns:
        Path to the unfinished checkpoint marker file.
    """
    marker_filepath = str(checkpoint_path).removesuffix(".nemo")
    marker_filepath = marker_filepath.removesuffix(".ckpt")
    marker_filepath = marker_filepath.removesuffix("-EMA")
    return Path(marker_filepath + UNFINISHED_CHECKPOINT_SUFFIX)


def remove_unfinished_checkpoints(checkpoint_dir: str):
    # Delete unfinished checkpoints from the filesystems.
    # "Unfinished marker" files are removed as well.

    checkpoint_dir = Path(checkpoint_dir)

    existing_marker_filepaths = {
        f.resolve() for f in checkpoint_dir.glob(f"*{UNFINISHED_CHECKPOINT_SUFFIX}") if f.is_file()
    }

    # some directories might be distributed checkpoints, we remove these if they have a unfinished marker
    all_dirpaths = {d.resolve() for d in checkpoint_dir.glob("*") if d.is_dir()}
    for ckpt_dirpath in all_dirpaths:
        possible_marker_path = format_checkpoint_unfinished_marker_path(ckpt_dirpath)
        if possible_marker_path in existing_marker_filepaths:
            print(f"Removing unfinished dist checkpoint: {ckpt_dirpath}")
            shutil.rmtree(ckpt_dirpath)

    # delete markers
    for marker_path in existing_marker_filepaths:
        os.remove(marker_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 remove_unfinished_checkpoints.py /path/to/ckpt/dir"
    remove_unfinished_checkpoints(sys.argv[1])
