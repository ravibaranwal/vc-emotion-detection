import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict,Any
from sklearn.ensemble import GradientBoostingClassifier
from src.data.data_ingestion import load_data,load_params,get_logger

logger = get_logger(__name__)


def _validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize model hyperparameters."""
    try:
        n_estimators = params["model_building"]["n_estimators"]
        learning_rate = params["model_building"]["learning_rate"]
    except KeyError as e:
        logger.error("Missing required key in params.yaml under model_building: %s", e)
        raise

    if not isinstance(n_estimators, (int, np.integer)) or int(n_estimators) <= 0:
        raise ValueError("model_building.n_estimators must be a positive integer.")
    n_estimators = int(n_estimators)

    if not isinstance(learning_rate, (float, int)) or float(learning_rate) <= 0:
        raise ValueError("model_building.learning_rate must be a positive number.")
    learning_rate = float(learning_rate)

    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    }

def _get_X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features matrix X and target vector y from a BoW DataFrame.
    Prefers 'sentiment' column as target; falls back to last column if needed.
    """
    if "sentiment" in df.columns:
        X = df.drop(columns=["sentiment"]).values
        y = df["sentiment"].values
    else:
        # fallback to last column
        X = df.iloc[:, 0:-1].values
        y = df.iloc[:, -1].values
        logger.warning("Target column 'sentiment' not found. Using last column as label.")
    return X, y

def dump_model(model_instance: Any, file_name: str) -> str:
    """
    Save model to <file_name>.pkl and return the full file path.
    """
    pkl_path = f"{file_name}.pkl"
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(model_instance, f)
        logger.info("Model successfully saved at %s", os.path.abspath(pkl_path))
    except Exception:
        logger.exception("Failed to save model to %s", pkl_path)
        raise
    return pkl_path


def main() -> None:

    try:
        params = load_params('params.yaml')
        model_cfg = _validate_model_params(params)

        logger.info("Training GradientBoostingClassifier with n_estimators=%d, learning_rate=%.4f",
                    model_cfg["n_estimators"], model_cfg["learning_rate"]
                    )

        # 2) Load training features
        train_csv = os.path.join("data", "processed", "train_bow.csv")
        train_data = load_data(train_csv)
        logger.info("Train features loaded: %s", train_data.shape)


        # 3) Extract X, y
        X_train, y_train = _get_X_y(train_data)
        logger.info("X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)

        # 4) Define and train model
        clf = GradientBoostingClassifier(
            n_estimators=model_cfg["n_estimators"],
            learning_rate=model_cfg["learning_rate"])
        
        clf.fit(X_train, y_train)
        logger.info("Model training completed.")

        # 6) Save model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model_gb")
        saved_path = dump_model(clf, model_path)
        logger.info("Model saved to %s", os.path.abspath(saved_path))

    except Exception:
        logger.exception("Model building pipeline failed.")
        raise


if __name__ == "__main__":
    main()