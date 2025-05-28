"""save the trained model to the specified directory."""

from datetime import datetime

from loguru import logger
from result import Ok, Result

from sentence_transformers import SentenceTransformer


def save_model(
    train_size: int,
    model_name: str,
    dataset_path: str,
    dataset_name: str,
    model: SentenceTransformer,
) -> Result[None, str]:
    """Save the trained model to a specified directory."""
    logger.info("Saving the trained model")
    model.save_pretrained("models/mpnet-base-all-nli-triplet/final")
    logger.info("Pushing the model to Hugging Face Hub")
    commit_message = (
        f"exec_date={datetime.now().isoformat()} -- "
        f"model_name={model_name} -- "
        f"dataset_path={dataset_path} -- "
        f"dataset_name={dataset_name} -- "
        f"train_size={train_size}"
    )
    model.push_to_hub(
        "mpnet-base-all-nli-triplet", exist_ok=True, commit_message=commit_message
    )
    return Ok(None)
