"""Multitask model."""

from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
import transformers


class MultitaskModel(transformers.PreTrainedModel):
    """Multitask model.

    This model inherits from [`transformers.PreTrainedModel`]

    """

    def __init__(
        self,
        encoder: Optional[Any],
        task_models_dict: Dict[Any, transformers.PreTrainedModel],
    ):
        """Init.

        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features.

        :param encoder: Encoder to share
        :type encoder: Optional[Any]
        :param task_models_dict: Dictionary
        :type task_models_dict: dict
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.task_models_dict = nn.ModuleDict(task_models_dict)

    @classmethod
    def create(
        cls,
        model_name: str,
        model_type_dict: Dict[
            Any, Type[transformers.AutoModelForSequenceClassification]
        ],
        model_config_dict: Dict[Any, transformers.PretrainedConfig],
    ) -> transformers.PreTrainedModel:
        """Create a MultitaskModel using the model class and config objects from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        :param model_name: Model name, the model id of a predefined model hosted inside a model repo on huggingface.co
        :type model_name: str
        :param model_type_dict: Model type dictionary
        :type model_type_dict: dict[str, transformers.PreTrainedModel]
        :param model_config_dict: Model type configuration
        :type model_config_dict: dict[str, transformers.PretrainedConfig]
        :return: A MultitaskModel
        :rtype: transformers.PreTrainedModel
        """
        shared_encoder = None
        task_models_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            task_models_dict[task_name] = model
        return cls(encoder=shared_encoder, task_models_dict=task_models_dict)

    @classmethod
    def get_encoder_attr_name(
        cls, model: transformers.AutoModelForSequenceClassification
    ) -> str:
        """Get encoder attribute name.

        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute

        :param model: Model
        :type model: transformers.PreTrainedModel
        :return: Name of encoder
        :rtype: str
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("DistilBert"):
            return "distilbert"
        elif model_class_name.startswith("T5"):
            return "encoder"
        elif model_class_name.startswith("Deberta"):
            return "deberta"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name: str, **kwargs: Any) -> Any:
        """Define the computation performed at every call.

        :param task_name: Task name
        :type task_name: str
        :param kwargs: Additional keyword arguments
        :type kwargs: additional keyword arguments
        :return: Output tensor
        :rtype: torch.Tensor
        """
        return self.task_models_dict[task_name](**kwargs)
