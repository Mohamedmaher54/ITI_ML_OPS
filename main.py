import hydra
from omegaconf import DictConfig
from src.logger import ExecutorLogger
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data
from src.training.train import encode_target_col, train_all_models

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = ExecutorLogger("training")

    # Extract config values
    dataset_name = cfg.process.file_name
    id_col = cfg.process.id_col
    target_col = cfg.process.target_col
    base_model_name = cfg.train.model_name

    logger.info("Starting training pipeline")

    # 1. Process raw data
    logger.info("Processing raw data")
    read_process_data(
        file_name=dataset_name,
        id_col=id_col,
        target_col=target_col,
        logger=logger,
        columns_to_drop=cfg.process.columns_to_drop,
        drop_missing_threshold=cfg.process.drop_missing_threshold
    )

    # 2. Encode and split
    logger.info("Encoding target variable")
    X_train, y_train, X_test, y_test = encode_target_col(
        file_name=dataset_name,
        target_col=target_col,
        model_name=base_model_name,
        logger=logger
    )

    # 3. Train models
    logger.info("Training models")
    train_all_models(X_train, y_train, base_model_name, logger)

    # 4. Evaluate each trained model
    logger.info("Evaluating models")
    for model_type in cfg.evaluate.models:
        evaluate(
            X_test=X_test,
            y_test=y_test,
            model_name=f"{base_model_name}_{model_type}",
            logger=logger
        )

    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()
