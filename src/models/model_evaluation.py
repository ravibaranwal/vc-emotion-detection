import numpy as np
import pandas as pd
import pickle
import os
import json
from typing import Dict,Any
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from src.data.data_ingestion import load_data, get_logger
from src.models.model_building import _get_X_y

logger = get_logger(__name__)

def load_model(model_path_no_ext: str):
    """
    Load a scikit-learn model saved as <model_path_no_ext>.pkl.
    Uses pickle for robustness (also works with pickle-generated files).
    """
    pkl_path = f"{model_path_no_ext}.pkl"
    if not os.path.exists(pkl_path):
        logger.error("Model file not found: %s", os.path.abspath(pkl_path))
        raise FileNotFoundError(f"Model file not found: {pkl_path}")
    logger.info("Loading model from %s", os.path.abspath(pkl_path))
    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

    except Exception:
        logger.exception("Failed to load model from %s", pkl_path)
        raise
    return model

def to_float(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Convert all metric values to plain float for JSON serialization."""
    return {k: float(v) for k, v in metrics.items()}


def main():
    try:
        # 1) Load Model
        clf = load_model(os.path.join("models", "model_gb"))

        # 2) Load features
        test_csv = os.path.join("data", "processed", "test_bow.csv")
        test_data = load_data(test_csv)
        logger.info("Test features loaded: %s", test_data.shape)

        # 3) Prepare X, y
        X_test, y_test = _get_X_y(test_data)
        logger.info("X_test shape: %s, y_test shape: %s", X_test.shape, y_test.shape)

        # 4) Predictions
        y_pred = clf.predict(X_test)

        # 5) Probabilities or scores for AUC metrics
        y_pred_proba = clf.predict_proba(X_test)[:,1]


        #calculate evaluation metrics
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)

        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }

        # 6) Save metrics as JSON
        out_dir = "reports"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_float(metrics_dict), f, indent=4)

        logger.info("Saved metrics to %s", os.path.abspath(out_path))

    except Exception:
        logger.exception("Evaluation pipeline failed.")
        raise


if __name__ == "__main__":
    main()