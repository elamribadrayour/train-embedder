"""main file for the train-embedder job."""

from typing import Annotated

from loguru import logger
from typer import Typer, Argument

from helpers import dataset, train, models, eval, save


app = Typer()


@app.command()
def run(
    train_size: Annotated[int, Argument(envvar="TRAIN_SIZE")],
    model_name: Annotated[str, Argument(envvar="MODEL_NAME")],
    model_device: Annotated[str, Argument(envvar="MODEL_DEVICE")],
    dataset_name: Annotated[str, Argument(envvar="DATASET_NAME")],
    dataset_path: Annotated[str, Argument(envvar="DATASET_PATH")],
) -> None:
    ds = dataset.get_dataset(
        path=dataset_path,
        name=dataset_name,
    ).unwrap()

    model = models.get_model(name=model_name, device=model_device).unwrap()
    loss = train.get_loss(model=model).unwrap()
    training_args = train.get_training_args().unwrap()
    evaluator = eval.get_eval_evaluator(ds=ds, model=model).unwrap()

    trainer = train.get_trainer(
        ds=ds,
        loss=loss,
        model=model,
        evaluator=evaluator,
        train_size=train_size,
        training_args=training_args,
    ).unwrap()

    logger.info("Starting training")
    trainer.train()  # type: ignore
    logger.info("Training completed")

    evaluator = eval.get_test_evaluator(ds=ds, model=model).unwrap()
    evaluator(model=model)

    save.save_model(
        model=model,
        train_size=train_size,
        model_name=model_name,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
    ).unwrap()


if __name__ == "__main__":
    app()
