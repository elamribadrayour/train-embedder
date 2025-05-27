"""save the trained model to the specified directory."""

import os
from datetime import datetime

from loguru import logger
from result import Ok, Result

from sentence_transformers import SentenceTransformer


def save_model(model: SentenceTransformer) -> Result[None, str]:
    """Save the trained model to a specified directory."""
    logger.info("Saving the trained model")
    model.save_pretrained("models/mpnet-base-all-nli-triplet/final")
    logger.info("Pushing the model to Hugging Face Hub")
    commit_message = (
        f"exec_date={datetime.now().isoformat()} -- "
        f"model_name={os.environ['MODEL_NAME']} -- "
        f"dataset_path={os.environ['DATASET_PATH']} -- "
        f"dataset_name={os.environ['DATASET_NAME']} -- "
        f"train_size={os.getenv('LIMIT', '0')}"
    )
    model.push_to_hub(
        "mpnet-base-all-nli-triplet", exist_ok=True, commit_message=commit_message
    )
    return Ok(None)
