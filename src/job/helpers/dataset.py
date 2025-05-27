"""dataset helpers for the job"""

import os

import datasets
from loguru import logger
from result import Ok, Result, Err


def get_dataset() -> Result[datasets.Dataset, str]:
    if "DATASET_NAME" not in os.environ or "DATASET_PATH" not in os.environ:
        return Err("Environment variables DATASET_NAME and DATASET_PATH must be set")
    name = os.environ["DATASET_NAME"]
    path = os.environ["DATASET_PATH"]
    dataset_id = f"{path}/{name}"
    logger.info(f"Loading dataset '{dataset_id}'")
    ds = datasets.load_dataset(path=path, name=name)
    return Ok(ds)


def get_train_dataset(
    ds: datasets.Dataset,
) -> Result[datasets.Dataset, str]:
    """Get the training dataset."""
    logger.info("Getting training dataset")
    nb_samples = int(os.getenv("LIMIT", 0))
    if nb_samples > 0:
        logger.info(f"Limiting training dataset to {nb_samples} samples")
        ds = ds["train"].select(range(nb_samples))
    else:
        logger.info("Using full training dataset")
        ds = ds["train"]
    return Ok(ds)


def get_eval_dataset(ds: datasets.Dataset) -> Result[datasets.Dataset, str]:
    logger.info("Getting eval dataset")
    return Ok(ds["dev"])


def get_test_dataset(ds: datasets.Dataset) -> Result[datasets.Dataset, str]:
    logger.info("Getting test dataset")
    return Ok(ds["test"])
