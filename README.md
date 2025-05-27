
# Train-Embedder

A Python project for fine-tuning sentence embedding models using the [Sentence Transformers](https://www.sbert.net/) library and Hugging Face Hub. This project is designed for training, evaluating, and versioning transformer-based models on triplet datasets, with support for Docker and `uv` for reproducible environments.

## Features

- Fine-tune transformer models (e.g., `microsoft/mpnet-base`) on triplet datasets.
- Evaluate models using triplet evaluation metrics.
- Save and version models locally and on the Hugging Face Hub.
- Modular codebase with helpers for training, evaluation, and saving.
- Docker and `uv` support for reproducible development and deployment.


## Quickstart

### 1. Install dependencies

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reproducible installs:

```sh
uv sync --frozen
```

### 2. Set environment variable

Create .env file with the required environment variables, you can take as an example the [.env.example](./.env.example)

### 2. Train and Evaluate

Run the main script to train and evaluate the model:

```sh
uv run src/job/main.py
```

### 3. Model Saving and Versioning

After training, the model is saved locally in `models/mpnet-base-all-nli-triplet/final` and pushed to the Hugging Face Hub.

## Customization

- **Dataset**: Modify `src/job/helpers/dataset.py` to change dataset loading or preprocessing.
- **Model**: Change the model name in `src/job/main.py` to use a different transformer.
- **Training Arguments**: Adjust hyperparameters in `SentenceTransformerTrainingArguments` in your training script.

## Requirements

- Python 3.12+
- [Sentence Transformers](https://www.sbert.net/)
- [datasets](https://github.com/huggingface/datasets)
- [loguru](https://github.com/Delgan/loguru)
- [result](https://github.com/dbrgn/result)
- [uv](https://github.com/astral-sh/uv) (optional, for dependency management)
- Docker (optional, for containerization)

## Hugging Face Hub Integration

The project supports pushing new versions of the model to the Hugging Face Hub. Each push creates a new commit/version. Make sure you set the HF_TOKEN with write access in the .env file.

## License

This project is licensed under the terms of the [WTFPL](./LICENSE).
