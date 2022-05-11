"""Model Operations.

This module contains operations connected with models.

"""
import transformers

from src.config import models_names
from src.models.MultitaskModel import MultitaskModel


def get_model(model_name: str, multitask: bool) -> transformers.PreTrainedModel:
    """Get model.

    :param model_name: Model name, the model id of a predefined model/tokenizer hosted inside a model repo on
    huggingface.co
    :type model_name: str
    :param multitask: Whether is multitask
    :type multitask: bool
    :return: Model
    :rtype: transformers.PreTrainedModel
    """
    if multitask:
        model = MultitaskModel.create(
            model_name=models_names[model_name],
            model_type_dict={
                "sentence": transformers.AutoModelForSequenceClassification,
                "text": transformers.AutoModelForSequenceClassification,
            },
            model_config_dict={
                "sentence": transformers.AutoConfig.from_pretrained(
                    models_names[model_name], num_labels=4
                ),
                "text": transformers.AutoConfig.from_pretrained(
                    models_names[model_name], num_labels=4
                ),
            },
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            models_names[model_name],
            config=transformers.AutoConfig.from_pretrained(
                models_names[model_name], num_labels=4
            ),
        )
    return model


def get_tokenizer(model_name: str) -> transformers.AutoTokenizer:
    """Get tokenizer.

    :param model_name: Model name, the model id of a predefined tokenizer hosted inside a model repo on
    huggingface.co.
    :type model_name: str
    :return: Tokenizer
    :rtype: transformers.AutoTokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(models_names[model_name])
    return tokenizer
