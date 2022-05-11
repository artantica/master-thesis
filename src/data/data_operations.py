"""Data Operations.

This module contains operations executed on dataset.

"""

import os
from typing import Any, Dict, List, Union

import datasets
import transformers
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from ..config import HOME_DIR, columns
from .utils import combine_files, convert_to_text_features, shorten_file

DatasetDictType = Dict[
    str, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
]


def fetch_data(
    task_names: List[Any], domain: str, language: str, shorten: bool = False
) -> DatasetDictType:
    """Fetch data from multiemo dataset after given `task_names` and load them to dictionary.

    :param task_names: List of task names
    :type task_names: List[str]
    :param domain: Review domain of texts, one of ["all", "hotels", "medicine", "products", "reviews", "Nhotels", "Nmadicine", "Nproducts", "Nreviews"]
    :type domain: str
    :param language: Language of texts
    :type language: str
    :param shorten: Flag to take shorten version of texts
    :type shorten: bool
    :return: Dataset dictionary, where keys are task names
    :rtype: DatasetDictType

    """
    dataset_dict = dict()
    for task_name in task_names:
        data_train, data_dev, data_test = get_data(task_name, domain, language, shorten)
        dataloader_path = os.path.join("src", "data", "multitask_dataset.py")

        dataset_dict[task_name] = load_dataset(
            dataloader_path,
            data_files={
                "train": data_train,
                "validation": data_dev,
                "test": data_test,
            },
        )
    return dataset_dict


def get_data(task_name: str, domain: str, language: str, shorten: bool) -> List[Any]:
    """Find file path for given parameters.

    :param task_name: List of task names
    :type task_name: str
    :param domain: Review domain of texts,
    :type domain: str
    :param language: Language of texts
    :type language: str
    :param shorten: Flag to take shorten version of texts
    :type shorten: bool
    :return: List of file paths in order `train`, `dev`, `test`
    :rtype: list[str]
    """
    filepaths = []
    _domain = domain

    for mode in ["train", "dev", "test"]:
        if domain.startswith("N") and mode == "test":
            _domain = domain[1:]

        filepath = os.path.join(
            HOME_DIR,
            "data",
            "multiemo",
            "multiemo2",
            f"{_domain}.{task_name}.{mode}.{language}.txt",
        )

        if task_name == "combined":
            if not os.path.exists(filepath):
                combine_files(mode=mode, domain=_domain, language=language)
        if shorten:
            shorten_filepath = os.path.join(
                HOME_DIR,
                "data",
                "multiemo",
                "multiemo2",
                f"{_domain}.{task_name}.{mode}.{language}_shorten.txt",
            )
            if not os.path.exists(shorten_filepath):
                shorten_file(org_filepath=filepath, new_filepath=shorten_filepath)
            filepath = shorten_filepath

        filepaths.append(filepath)
        _domain = domain
    return filepaths


def preprocess_dataset(
    dataset_dict: Dict[str, Dict[str, datasets.arrow_dataset.Dataset]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, Dict[str, datasets.arrow_dataset.Dataset]]:
    """Preprocess dataset.

    :param dataset_dict: Dataset dictionary
    :type dataset_dict: Dict[str, Dict[str, datasets.arrow_dataset.Dataset]]
    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param max_length: Max length
    :type max_length: int
    :return: Preprocess dataset dictionary
    :rtype: Dict[str, Dict[str, datasets.arrow_dataset.Dataset]]
    """
    features_dict: Dict[str, Dict[str, datasets.arrow_dataset.Dataset]] = dict()
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                lambda example: convert_to_text_features(
                    tokenizer=tokenizer, example_batch=example, max_length=max_length
                ),
                batched=True,
                load_from_cache_file=False,
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns,
            )
    return features_dict


def get_datasets(
    features_dict: Dict[str, Dict[str, datasets.arrow_dataset.Dataset]], multitask: bool
) -> tuple:
    """Get dataset splits.

    :param features_dict:
    :type features_dict: medicine
    :param multitask: Whether is multitask dataset
    :type multitask: bool
    :return: Train, dev and test datasets
    :rtype: tuple
    """
    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }
    eval_dataset = {
        task_name: dataset["validation"] for task_name, dataset in features_dict.items()
    }
    test_dataset_temp = {
        task_name: dataset["test"] for task_name, dataset in features_dict.items()
    }
    if not multitask:
        return train_dataset, eval_dataset, test_dataset_temp

    test_dataset = {
        "sentence": {"sentence": test_dataset_temp["sentence"]},
        "text": {"text": test_dataset_temp["text"]},
    }
    return train_dataset, eval_dataset, test_dataset
