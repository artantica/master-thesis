"""Main module for experiments."""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import click

from src.config import DATA_FORMAT, HOME_DIR, MODELS_DIR, models_names
from src.data.data_operations import fetch_data, get_datasets, preprocess_dataset
from src.models.model_operations import get_model, get_tokenizer
from src.train import get_trainer, predict, save_predictions, train
from src.utils import get_checkpoint

logging.basicConfig(level=logging.INFO)


def save_config(config: Dict[str, Union[int, str, bool, float]], name: str) -> None:
    """Save all parameters for experiments to file.

    :param config: Configuration dictionary
    :type config:
    :param name: Name of experiments
    :type name: str
    """
    path = os.path.join(HOME_DIR, MODELS_DIR, name)
    Path(path).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(path, "script_config.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        print("File", file_path, "was created.")


@click.command()
@click.option(
    "--checkpoint-date",
    default=None,
    type=str,
    help=f"Date for model to continue in format {DATA_FORMAT}.",
)
@click.option(
    "--task-name",
    required=True,
    prompt="Choose task name:",
    type=click.Choice(["text", "sentence", "both", "combined"]),
    help="Task name.",
)
@click.option(
    "--domain",
    prompt="Choose domain:",
    required=True,
    type=click.Choice(
        [
            "all",
            "hotels",
            "medicine",
            "products",
            "reviews",
            "Nhotels",
            "Nmadicine",
            "Nproducts",
            "Nreviews",
        ]
    ),
    help="Name of the model, where `N[domain]` means all domain except [domain].",
)
@click.option(
    "--language",
    default="en",
    type=click.Choice(["en", "de", "es", "fr", "it", "ja", "nl", "pl", "ru", "zh"]),
    help="Language of data.",
)
@click.option(
    "--shorten", is_flag=True, show_default=True, default=False, help="Small datasets."
)
@click.option(
    "--model-name",
    prompt="Choose model name:",
    required=True,
    type=click.Choice(list(models_names.keys())),
    help="Name of the model.",
)
@click.option("--max-length", type=int, default=128, help="Max length")
@click.option(
    "--learning-rate",
    type=float,
    default=1e-5,
    help=" The initial learning rate for AdamW optimizer.",
)
@click.option(
    "--num-train-epochs",
    type=int,
    default=3,
    help="Total number of training epochs to perform.",
)
@click.option(
    "--per-device-train-batch-size",
    type=int,
    default=4,
    help="The batch size per GPU/TPU core/CPU for training.",
)
@click.option(
    "--per-device-eval-batch-size",
    type=int,
    default=4,
    help=" The batch size per GPU/TPU core/CPU for evaluation.",
)
@click.option(
    "--save-steps",
    type=int,
    default=400,
    help="Number of updates steps before two checkpoint saves.",
)
@click.option(
    "--save-total-limit",
    type=int,
    default=2,
    help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints "
    "'in output_dir'.",
)
@click.option(
    "--eval-steps",
    type=int,
    default=200,
    help="Number of update steps between two evaluations.",
)
@click.option(
    "--metric-for-best-model",
    type=bool,
    default=True,
    help="f1_macro",
)
def run(
    task_name: str,
    domain: str,
    model_name: str,
    language: str = None,
    shorten: bool = False,
    max_length: int = 128,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    save_steps: int = 2000,
    save_total_limit: int = 5,
    eval_steps: int = 1000,
    checkpoint_date: str = None,
    metric_for_best_model: bool = True,
) -> None:
    """Run experiments.

    :param task_name: Task name, available choices are ('text', 'sentence', 'both', 'combined')
    :type task_name: str
    :param domain: Review domain of texts
    :type domain: str
    :param model_name: Model name, available choices are
    :type model_name: str
    :param language: Language of texts
    :type language: str
    :param shorten: Whether it should use shorten files
    :type shorten: bool
    :param max_length: Max length
    :type max_length: int
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
    :param checkpoint_date: Checkpoint date
    :type checkpoint_date: str
    """
    config = {
        "task_name": task_name,
        "domain": domain,
        "model_name": model_name,
        "language": language,
        "shorten": shorten,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "eval_steps": eval_steps,
        "checkpoint_date": checkpoint_date,
        "metric_for_best_model": metric_for_best_model,
    }
    print(",".join(str(x) for x in config.keys()))
    exit()
    task_names = None
    if task_name in ["sentence", "text", "combined"]:
        task_names = [task_name]
    elif task_name == "both":
        task_names = ["sentence", "text"]

    multitask = False if len(task_names) == 1 else True

    date = (
        datetime.now().strftime(DATA_FORMAT)
        if checkpoint_date is None
        else checkpoint_date
    )
    name = f"{model_name}_{domain}_{task_name}_{date}"
    save_config(config=config, name=name)

    dataset_dict = fetch_data(
        task_names=task_names, domain=domain, language=language, shorten=shorten
    )

    model = get_model(model_name, multitask)
    tokenizer = get_tokenizer(model_name)

    features_dict = preprocess_dataset(dataset_dict, tokenizer, max_length)

    train_dataset, eval_dataset, test_dataset = get_datasets(features_dict, multitask)

    trainer, training_time = get_trainer(
        name,
        multitask,
        model,
        train_dataset,
        eval_dataset,
        learning_rate,
        num_train_epochs,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        save_steps,
        save_total_limit,
        eval_steps,
        metric_for_best_model,
    )

    checkpoint = get_checkpoint(name) if checkpoint_date else None

    trainer = train(trainer, checkpoint)

    # save_best_model(tokenizer, trainer, name)

    prediction_dict = predict(task_names, trainer, test_dataset)
    prediction_dict["training_time"] = training_time

    save_predictions(prediction_dict, name)


if __name__ == "__main__":
    run()
