"""main file for the train-embedder job."""

from loguru import logger
from result import Result, Ok
from dotenv import load_dotenv

from helpers import dataset, train, models, eval, save


def main() -> Result[None, str]:
    load_dotenv()

    ds = dataset.get_dataset().unwrap()

    model = models.get_model().unwrap()
    callbacks = train.get_callbacks().unwrap()
    loss = train.get_loss(model=model).unwrap()
    training_args = train.get_training_args().unwrap()
    evaluator = eval.get_eval_evaluator(ds=ds, model=model).unwrap()

    trainer = train.get_trainer(
        ds=ds,
        loss=loss,
        model=model,
        evaluator=evaluator,
        callbacks=callbacks,
        training_args=training_args,
    ).unwrap()

    logger.info("Starting training")
    trainer.train()  # type: ignore
    logger.info("Training completed")

    evaluator = eval.get_test_evaluator(ds=ds, model=model).unwrap()
    evaluator(model=model)

    save.save_model(model=model).unwrap()
    return Ok(None)


if __name__ == "__main__":
    main().unwrap()
