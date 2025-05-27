"""evaluation helpers for the training process."""

from loguru import logger
from result import Ok, Result

import datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator

from helpers import dataset


def get_eval_evaluator(
    ds: datasets.Dataset,
    model: SentenceTransformer,
) -> Result[TripletEvaluator, str]:
    """Get the evaluator for the training process."""
    logger.info("Loading evaluator")
    eval_dataset = dataset.get_eval_dataset(ds=ds).unwrap()
    eval_evaluator = TripletEvaluator(
        name="all-nli-eval",
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )
    eval_evaluator(model=model)
    return Ok(eval_evaluator)


def get_test_evaluator(
    ds: datasets.Dataset, model: SentenceTransformer
) -> Result[TripletEvaluator, str]:
    """Get the evaluator for the test dataset."""
    logger.info("Loading test dataset and evaluator")
    test_ds = dataset.get_test_dataset(ds=ds).unwrap()
    test_evaluator = TripletEvaluator(
        name="all-nli-test",
        anchors=test_ds["anchor"],
        positives=test_ds["positive"],
        negatives=test_ds["negative"],
    )
    return Ok(test_evaluator)
