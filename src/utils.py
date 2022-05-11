"""Utils."""
import os
import re
from pathlib import Path


def get_checkpoint(name: str) -> str:
    """Get path to last checkpoint.

    :param name: Name of experiment
    :type name: str
    :return: Path to last checkpoint
    :rtype: str
    """
    directory = os.path.join("models", name)

    checkpoints = []
    for filename in os.listdir(directory):
        path = Path(os.path.join(directory, filename))
        if path.is_dir():
            if filename.startswith("checkpoint"):
                checkpoint = int(re.findall("[0-9]+", filename)[0])
                checkpoints.append(checkpoint)

    last_checkpoint = sorted(checkpoints)[-1]
    return os.path.join(directory, f"checkpoint-{last_checkpoint}")
