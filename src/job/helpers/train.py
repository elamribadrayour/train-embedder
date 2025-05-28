"""training helpers for the SentenceTransformer model."""

from loguru import logger
from result import Ok, Result

import datasets
from sentence_transformers import (
    util,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

from helpers import dataset


def get_loss(
    model: SentenceTransformer,
) -> Result[MultipleNegativesRankingLoss, str]:
    """Get the loss function for training."""
    loss = MultipleNegativesRankingLoss(
        model=model, scale=20.0, similarity_fct=util.cos_sim
    )
    return Ok(loss)


def get_training_args() -> Result[SentenceTransformerTrainingArguments, str]:
    """Get the training arguments for the SentenceTransformer model."""
    logger.info("Loading training arguments")
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models",
        dataloader_pin_memory=False,
        # training parameters (Optional)
        warmup_ratio=0.1,
        num_train_epochs=1,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        fp16=False,  # Set to False if your GPU can't handle FP16
        bf16=False,  # Set to True if your GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
        # tracking/debugging parameters (Optional)
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        save_strategy="steps",
    )
    return Ok(args)


def get_trainer(
    train_size: int,
    ds: datasets.Dataset,
    model: SentenceTransformer,
    evaluator: TripletEvaluator,
    loss: MultipleNegativesRankingLoss,
    training_args: SentenceTransformerTrainingArguments,
) -> Result[SentenceTransformerTrainer, str]:
    """Get the SentenceTransformer trainer."""
    logger.info("Loading SentenceTransformer trainer")
    eval_dataset = dataset.get_eval_dataset(ds=ds).unwrap()
    train_dataset = dataset.get_train_dataset(ds=ds, train_size=train_size).unwrap()

    trainer = SentenceTransformerTrainer(
        loss=loss,
        model=model,
        args=training_args,
        evaluator=evaluator,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
    )
    return Ok(trainer)
