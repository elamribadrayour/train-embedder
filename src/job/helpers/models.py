"""model definition for the train-embedder job."""

from loguru import logger
from result import Ok, Result

from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData


def get_model(name: str, device: str) -> Result[SentenceTransformer, str]:
    """Get the SentenceTransformer model."""
    logger.info(f"Loading SentenceTransformer model: {name} on device: {device}")
    model = SentenceTransformer(
        device=device,
        model_name_or_path=name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="wtfpl",
            model_name=name,
            tags=["sentence-transformers", "triplet-loss", "nli", "tutorial"],
        ),
    )
    return Ok(model)
