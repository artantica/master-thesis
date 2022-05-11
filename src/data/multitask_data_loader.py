"""Data loader for multitasking.

Data loader that combines and samples from multiple single-task
data loaders.

"""
from typing import Any, Generator

import numpy as np


class MultitaskDataloader:
    """Custom multitask data loader.

    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict: dict) -> None:
        """Init.

        :param dataloader_dict: Data loader dictionary
        :type dataloader_dict: dict
        """
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self) -> int:
        """Len of data loader.

        :return: Len of all batches
        :rtype: Generator[Any, Any, Any]
        """
        return sum(self.num_batches_dict.values())

    def __iter__(self) -> Generator[dict, Any, Any]:
        """Iterate over task names.

        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.

        :return: Batch with dictionary of `text` and `label`
        :rtype: Generator[dict, Any, Any]:
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]

        np_task_choice_list = np.array(task_choice_list)
        np.random.shuffle(np_task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in np_task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])
