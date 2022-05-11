"""Train and evaluate."""
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
import transformers

from src.data.multitask_data_collator import MultitaskDataCollator
from src.metrics import compute_metrics
from src.multitask_trainer import MultitaskTrainer
from src.numpy_encoder import NumpyEncoder


def get_trainer(
    name: str,
    multitask: bool,
    model: transformers.PreTrainedModel,
    train_dataset: Dict[str, Optional[torch.utils.data.dataset.Dataset[Any]]],
    eval_dataset: Dict[str, Optional[torch.utils.data.dataset.Dataset[Any]]],
    learning_rate: float = 1e-5,
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    save_steps: int = 2000,
    save_total_limit: int = 5,
    eval_steps: int = 1000,
    metric_for_best_model: bool = False,
) -> transformers.Trainer:
    """Get trainer.

    :param name: Name of experiment
    :type name: str
    :param multitask: Whether is multitask
    :type multitask: bool
    :param model: Model
    :type model: transformers.PreTrainedModel
    :param train_dataset: Train dataset
    :type train_dataset: Dict[str, datasets.arrow_dataset.Dataset]
    :param eval_dataset: Evaluation dataset
    :type eval_dataset: Dict[str, datasets.arrow_dataset.Dataset]
    :param learning_rate: Learning rate
    :type learning_rate: float
    :param num_train_epochs: Number of epochs
    :type num_train_epochs: int
    :param per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training.
    :type per_device_train_batch_size: int
    :param per_device_eval_batch_size: The batch size per GPU/TPU core/CPU for evaluation.
    :type per_device_eval_batch_size: int
    :param save_steps: Number of updates steps before two checkpoint
    :type save_steps: int
    :param save_total_limit: Limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
    :type save_total_limit: int
    :param eval_steps: Number of update steps between two evaluations
    :type eval_steps: int
    :return: Trainer
    :rtype: transformers.Trainer
    """
    if metric_for_best_model:
        training_args = transformers.TrainingArguments(
            output_dir=os.path.join("models", name),
            # overwrite_output_dir=True,
            learning_rate=learning_rate,
            do_train=True,
            do_eval=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=1000,
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_strategy=transformers.IntervalStrategy.STEPS,
            save_total_limit=save_total_limit,
            logging_dir="./logs",
            evaluation_strategy=transformers.IntervalStrategy.STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to=["wandb"],
            run_name=f"{name}",
        )
    else:
        training_args = transformers.TrainingArguments(
            output_dir=os.path.join("models", name),
            # overwrite_output_dir=True,
            learning_rate=learning_rate,
            do_train=True,
            do_eval=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=1000,
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_strategy=transformers.IntervalStrategy.STEPS,
            save_total_limit=save_total_limit,
            logging_dir="./logs",
            evaluation_strategy=transformers.IntervalStrategy.STEPS,
            load_best_model_at_end=True,
            report_to=["wandb"],
            run_name=f"{name}",
        )

    if multitask:
        trainer = MultitaskTrainer(
            model=model,
            args=training_args,
            data_collator=MultitaskDataCollator(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        level = list(train_dataset.keys())[0]
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset[level],
            eval_dataset=eval_dataset[level],
            compute_metrics=compute_metrics,
        )
    return trainer


def predict(
    task_names: List[str],
    trainer: transformers.Trainer,
    test_dataset: Dict[str, datasets.arrow_dataset.Dataset],
) -> Dict[str, Any]:
    """Predict.

    :param task_names: List of task names
    :type task_names: list[str]
    :param trainer: Trainer
    :type trainer: transformers.Trainer
    :param test_dataset:
    :type test_dataset: datasets.arrow_dataset.Dataset
    :return: Prediction dictionary
    :rtype: Dict[str, transformers.trainer_utils.PredictionOutput]
    """
    prediction_dict: Dict[str, transformers.trainer_utils.PredictionOutput] = dict()

    for task_name in task_names:
        prediction_dict[task_name] = trainer.predict(
            test_dataset[task_name],
        )

    return prediction_dict


def save_predictions(
    prediction_dict: Dict[
        str, Union[float, transformers.trainer_utils.PredictionOutput]
    ],
    name: str,
) -> None:
    """Save predictions.

    :param prediction_dict: Prediction dictionary to save
    :type prediction_dict: Dict[str, Union[float, transformers.trainer_utils.PredictionOutput]]
    :param name: Name of experiment
    :type name: str
    """
    json_string = (
        json.dumps(prediction_dict, indent=2, sort_keys=True, cls=NumpyEncoder) + "\n"
    )
    json_path = os.path.join("models", name, "test_json.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_string)


def train(
    trainer: transformers.Trainer, checkpoint_name: str = None
) -> transformers.Trainer:
    """Train.

    :param trainer: Trainer
    :type trainer: transformers.Trainer
    :param checkpoint_name: Checkpoint name, eg. `checkpoint-<number>`
    :type checkpoint_name: str
    :return: Trainer
    :rtype: transformers.Trainer
    """
    start_time = time.time()
    if checkpoint_name:
        trainer.train(checkpoint_name)
    else:
        trainer.train()
    training_time = time.time() - start_time
    trainer.save_model()
    return trainer, training_time


def save_best_model(
    tokenizer: transformers.PreTrainedTokenizer,
    trainer: transformers.Trainer,
    name: str,
) -> None:
    """Save best model.

    :param tokenizer: Tokenizer
    :type tokenizer: transformers.PreTrainedTokenizer
    :param trainer: Trainer
    :type trainer: transformers.Trainer
    :param name: Name of experiment
    :type name: str
    """
    path = os.path.join("models", name, "final")

    tokenizer.save_pretrained(path)
    trainer.model.save_pretrained(path)
