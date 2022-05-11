"""Multitask DataCollator.

A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.

"""
from typing import Any, Dict, List

import torch
from transformers.data.data_collator import DefaultDataCollator


class MultitaskDataCollator(DefaultDataCollator):
    """Custom data collator.

    Extending the existing DataCollator to work with NLP dataset batches.
    Simply collates batches of dict-like objects and performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    """

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors: str = None
    ) -> Dict[str, Any]:
        """Call method.

        :param features: List of samples from a Dataset
        :type features:  List[Dict[str, Any]]
        :param return_tensors: The type of Tensor to return. Allowable values are "np", "pt" and "tf".
        :type return_tensors: str
        :return: Dictionary of PyTorch/TensorFlow tensors
        :rtype: Dict[str, Any]
        """
        first = features[0]
        batch = {}

        if isinstance(first, dict):

            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.float
                    )
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            return DefaultDataCollator().__call__(features)
