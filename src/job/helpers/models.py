"""model definition for the train-embedder job."""

import os

from loguru import logger
from result import Ok, Result, Err

from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData


def get_model() -> Result[SentenceTransformer, str]:
    """Get the SentenceTransformer model."""
    if "MODEL_NAME" not in os.environ:
        return Err("Environment variable MODEL_NAME must be set")

    model_id = os.environ["MODEL_NAME"]
    logger.info(f"Loading SentenceTransformer model: {model_id}")
    model = SentenceTransformer(
        model_name_or_path=model_id,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="wtfpl",
            model_name=model_id,
            tags=["sentence-transformers", "triplet-loss", "nli", "tutorial"],
        ),
    )
    return Ok(model)
