"""dataset helpers for the job"""

import datasets
from loguru import logger
from result import Ok, Result


def get_dataset(name: str, path: str) -> Result[datasets.Dataset, str]:
    dataset_id = f"{path}/{name}"
    logger.info(f"Loading dataset '{dataset_id}'")
    ds = datasets.load_dataset(path=path, name=name)
    return Ok(ds)


def get_train_dataset(
    ds: datasets.Dataset,
    train_size: int,
) -> Result[datasets.Dataset, str]:
    """Get the training dataset."""
    logger.info("Getting training dataset")
    ds = ds["train"] if train_size == 0 else ds["train"].select(range(train_size))
    return Ok(ds)


def get_eval_dataset(ds: datasets.Dataset) -> Result[datasets.Dataset, str]:
    logger.info("Getting eval dataset")
    return Ok(ds["dev"])


def get_test_dataset(ds: datasets.Dataset) -> Result[datasets.Dataset, str]:
    logger.info("Getting test dataset")
    return Ok(ds["test"])
