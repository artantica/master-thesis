"""Custom data loader.

Data loader with task names to support multitask learning.

"""
from typing import Any, Generator

from torch.utils.data.dataloader import DataLoader

from .str_ignore_device import StrIgnoreDevice


class DataLoaderWithTaskName:
    """Wrapper around a DataLoader.

    Solution for `MultitaskDataloader` to also yield a task name.
    """

    def __init__(self, task_name: str, data_loader: DataLoader) -> None:
        """Init.

        :param task_name: Task name
        :type task_name: str
        :param data_loader: Data loader
        :type data_loader: Dataloader
        """
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self) -> int:
        """Length of data loader.

        :return: Length of data loader
        :rtype: Generator[int]
        """
        return len(self.data_loader)

    def __iter__(self) -> Generator[Any, Any, Any]:
        """Iterate over batches.

        :return: Batch with dictionary of `text` and `label`
        :rtype: Generator[Any, Any, Any]
        """
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch
