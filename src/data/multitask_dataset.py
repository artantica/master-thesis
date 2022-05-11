"""MultitaskDataset."""
from typing import Any, Generator, List

import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)


class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Dataset."""

    def __init__(self, **kwargs):
        """Builder config for MultitaskDataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)


class MultitaskDataset(datasets.GeneratorBasedBuilder):
    """Class for datasets with data generation based on dict generators.

    `MultitaskDataset` is a convenience class that abstracts away much
    of the data writing and reading of `DatasetBuilder`.

    """

    BUILDER_CONFIGS = [
        DatasetConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    labels_dict = {
        "meta_zero": 0,
        "meta_minus_m": 1,
        "meta_plus_m": 2,
        "meta_amb": 3,
        "z_zero": 0,
        "z_minus_m": 1,
        "z_plus_m": 2,
        "z_amb": 3,
    }

    def _info(self) -> datasets.DatasetInfo:
        """Construct the DatasetInfo object.

        :return: The dataset information
        :rtype: datasets.DatasetInfo
        """
        return datasets.DatasetInfo(
            description="Multitask dataset",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Specify feature dictionary generators and dataset splits.

        :param dl_manager: Download manager to download the data
        :type dl_manager: datasets.DownloadManager
        :return: List of `SplitGenerator`
        :rtype: list[SplitGenerator]
        """
        downloaded_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=str(datasets.Split.TRAIN),
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.VALIDATION),
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.TEST),
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(
        self, **kwargs: Any
    ) -> Any:  # Generator[list[Any], None, None]:
        """Get the examples in the raw (text) form.

        This function preprocess the examples from the raw data to the preprocessed
        dataset files.

        :param filepath: The file path to dataset file
        :type filepath: str
        :return:  Feature dictionary ready to be encoded and written to disk
        :rtype: Generator[list[Any]]
        """
        filepath = kwargs["filepath"]

        logger.info("generating examples from = %s", filepath)
        if type(filepath) is list:
            logger.info("file path is list. taking %s", filepath[0])
            filepath = filepath[0]

        with open(filepath, "r") as file:
            lines = file.readlines()
        data = []

        for line in lines:
            index = line.find("__label__")
            text = line[:index].strip()
            label = line[index + 9 :].strip()
            data.append((text, label))

        df = pd.DataFrame(data, columns=["text", "label"])
        # Encode labels in column
        df["labels"] = df.apply(lambda row: self.labels_dict[row["label"]], axis=1)

        for idx, row in df.iterrows():
            yield idx, {
                "text": row["text"],
                "label": row["labels"],
            }
