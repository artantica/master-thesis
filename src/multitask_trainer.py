"""Multitask Trainer.

The MultitaskTrainer class, to easily finetune Transformers on a new task.
"""
from typing import Any, Optional

import datasets
import torch
import transformers
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.utils import is_torch_tpu_available

if is_torch_tpu_available():
    import torch_xla.distributed.parallel_loader as pl

from .data.data_loader_with_task_name import DataLoaderWithTaskName
from .data.multitask_data_loader import MultitaskDataloader


class MultitaskTrainer(transformers.Trainer):
    """Multitask trainer.

    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
    """

    def get_single_train_dataloader(
        self, task_name: str, train_dataset: Any
    ) -> DataLoaderWithTaskName:
        """Create a single-task train data loader that also yields task names.

        :param task_name: Task name
        :type task_name: str
        :param train_dataset: Train dataset
        :type train_dataset: torch.utils.data.dataset.Dataset[Any]
        :return: Data loader
        :rtype: DataLoaderWithTaskName

        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(train_dataset)
        else:
            train_sampler = (
                RandomSampler(train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(train_dataset)
            )

        data_loader = DataLoaderWithTaskName(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        if is_torch_tpu_available():
            data_loader = pl.ParallelLoader(
                data_loader, [self.args.device]
            ).per_device_loader(self.args.device)
        return data_loader

    def get_single_eval_dataloader(
        self, task_name: str, eval_dataset: Any
    ) -> DataLoaderWithTaskName:
        """Create a single-task evaluation data loader that also yields task names.

        :param task_name: Task name
        :type task_name: str
        :param eval_dataset: Train dataset
        :type eval_dataset: torch.utils.data.dataset.Dataset[Any]
        :return: Data loader
        :rtype: DataLoaderWithTaskName
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")

        else:
            eval_sampler = (
                RandomSampler(eval_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(eval_dataset)
            )

        data_loader = DataLoaderWithTaskName(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=self.data_collator,
            ),
        )

        return data_loader

    def get_single_test_dataloader(self, task_name: str, test_dataset: Any):
        """Create a single-task test data loader that also yields task names.

        :param task_name: Task name
        :type task_name: str
        :param test_dataset: Train dataset
        :type test_dataset: torch.utils.data.dataset.Dataset[Any]
        :return: Data loader
        :rtype: DataLoaderWithTaskName
        """
        if self.test_dataset is None:
            raise ValueError("Trainer: test requires a test_dataset.")

        else:
            test_sampler = (
                RandomSampler(test_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(test_dataset)
            )

        data_loader = DataLoaderWithTaskName(
            task_name=task_name,
            data_loader=DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=test_sampler,
                collate_fn=self.data_collator,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """Get train dataloader.

        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def get_eval_dataloader(
        self, eval_dataset: Optional[torch.utils.data.Dataset] = None
    ):
        """Get evaluation dataloader.

        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        self.label_names = ["labels"]
        return MultitaskDataloader(
            {
                task_name: self.get_single_eval_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.eval_dataset.items()
            }
        )

    def get_test_dataloader(self, test_dataset: torch.utils.data.Dataset):
        """Get test dataloader.

        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader.

        :param test_dataset: Test dataset
        :type test_dataset: datasets.arrow_dataset.Dataset
        :return:
        :rtype:
        """
        self.label_names = ["labels"]
        return MultitaskDataloader(
            {
                task_name: self.get_single_eval_dataloader(task_name, task_dataset)
                for task_name, task_dataset in test_dataset.items()
            }
        )
