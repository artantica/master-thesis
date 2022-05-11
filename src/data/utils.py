"""Utils for data module."""
import os

import datasets
import transformers

from ..config import HOME_DIR


def shorten_file(
    org_filepath: str, new_filepath: str, number_of_lines: int = 10
) -> None:
    """Shorten file.

    Take only first `number_of_lines` lines to new shorten file.

    :param org_filepath: Path of original file
    :type org_filepath: str
    :param new_filepath: Path to shorten file
    :type new_filepath: str
    :param number_of_lines: Number of lines to shorten file
    :type number_of_lines: int
    """
    with open(org_filepath, "r") as file:
        lines = file.readlines()

    data = []
    for idx in range(number_of_lines):
        data.append(lines[idx])

    with open(new_filepath, "w+") as file:
        file.writelines(data)


def combine_files(mode: str, domain: str, language: str) -> None:
    """Combine two files.

    :param mode: Mode of dataset file. Available choices: (`train`, `dev`, `test`)
    :type mode: str
    :param domain: Review domain of texts
    :type domain: str
    :param language: Language of texts
    :type language: str
    """
    all_lines = []
    for task_name in ["text", "sentence"]:
        filepath = os.path.join(
            HOME_DIR,
            "data",
            "multiemo",
            "multiemo2",
            f"{domain}.{task_name}.{mode}.{language}.txt",
        )

        with open(filepath, "r", encoding="utf-8") as file:  # TODO: check encoding
            lines = file.readlines()
        all_lines.extend(lines)

    filepath = os.path.join(
        HOME_DIR,
        "data",
        "multiemo",
        "multiemo2",
        f"{domain}.combined.{mode}.{language}.txt",
    )
    with open(filepath, "w+", encoding="utf-8") as file:
        file.writelines(all_lines)


def convert_to_text_features(
    tokenizer: transformers.PreTrainedTokenizer,
    example_batch: datasets.arrow_dataset.Dataset,
    max_length: int = 512,
) -> datasets.arrow_dataset.Dataset:
    """Convert from raw text to tokenized text inputs.

    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param example_batch: Batch
    :type example_batch: datasets.arrow_dataset.Dataset
    :param max_length: Max length
    :type max_length: int
    :return: Dataset with tokenized text inputs
    :rtype: datasets.arrow_dataset.Dataset
    """
    inputs = list(example_batch["text"])
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features
