"""Metrics evaluation."""
from typing import Any, Dict

import transformers
from datasets import load_metric

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")


def compute_metrics(eval_pred: transformers.EvalPrediction) -> Dict[Any, Any]:
    """Compute metrics at evaluation.

    param eval_pred: Evaluation output
    :type eval_pred: transformers.EvalPrediction
    :return: Computed metrics
    :rtype: dict
    """
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions.argmax(-1)

    f1 = f1_metric.compute(predictions=predictions, references=labels, average=None)
    f1["f1"] = f1["f1"].tolist()

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)[
            "accuracy"
        ],
        "f1_weighted": f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"],
        "f1_micro": f1_metric.compute(
            predictions=predictions, references=labels, average="micro"
        )["f1"],
        "f1_macro": f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"],
        "f1": f1["f1"],
        "precision_weighted": precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"],
        "precision_macro": precision_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["precision"],
        "recall_weighted": recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"],
        "recall_macro": recall_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["recall"],
    }
