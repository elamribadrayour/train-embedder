"""Errors definitions"""


class DatasetNotLoaded(Exception):
    """Raised when a dataset is not found."""

    def __init__(self, dataset_id: str, err: Exception):
        super().__init__(f"Dataset with ID '{dataset_id}' not found: error={err}")


class ModelNotLoaded(Exception):
    """Raised when a model is not found."""

    def __init__(self, model_id: str, err: Exception):
        super().__init__(f"Model with ID '{model_id}' not found: error={err}")


class LossNotLoaded(Exception):
    """Raised when a loss function is not found."""

    def __init__(self, loss_id: str, err: Exception):
        super().__init__(f"Loss function with ID '{loss_id}' not found: error={err}")


class TrainingArgsNotLoaded(Exception):
    """Raised when training arguments are not found."""

    def __init__(self, args_id: str, err: Exception):
        super().__init__(
            f"Training arguments with ID '{args_id}' not found: error={err}"
        )


class EvaluatorNotLoaded(Exception):
    """Raised when an evaluator is not found."""

    def __init__(self, evaluator_id: str, err: Exception):
        super().__init__(f"Evaluator with ID '{evaluator_id}' not found: error={err}")


class TrainerNotLoaded(Exception):
    """Raised when a trainer is not found."""

    def __init__(self, trainer_id: str, err: Exception):
        super().__init__(f"Trainer with ID '{trainer_id}' not found: error={err}")
