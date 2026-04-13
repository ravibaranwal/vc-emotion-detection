import numpy as np
import pandas as pd
import os
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Any


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger
    
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"app_{timestamp}.log")

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = get_logger(__name__)

def load_params(params_path: str) -> Dict[str,Any]:
    
    """
    Load a YAML parameter file and return it as a dictionary.

    """
    
    if not params_path:
        logger.error("Params path is not provided.")
        raise ValueError("params_path must be a non-empty string.")

    abs_path = os.path.abspath(params_path)
    logger.debug("Loading params file from %s", abs_path)

    if not os.path.exists(params_path):
        logger.error("Params file not found: %s", params_path)
        raise FileNotFoundError(f"Params file not found: {params_path}")

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        logger.exception("Invalid YAML in params file: %s", params_path)
        raise
    except Exception:
        logger.exception("Unexpected error reading params file: %s", params_path)
        raise

    if data is None:
        logger.error("Params file is empty or contains only nulls: %s", params_path)
        raise ValueError(f"Params file is empty: {params_path}")

    if not isinstance(data, dict):
        logger.error("Params file must deserialize to a dict, got %s", type(data).__name__)
        raise TypeError("Params must be a dictionary at the top level.")

    logger.debug("Params loaded successfully with keys: %s", list(data.keys()))
    return data

    
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load a CSV file from the given path and return it as a pandas DataFrame.

    """
    if not data_path:
        logger.error("Data path is not provided.")
        raise ValueError("data_path must be a non-empty string.")        

    abs_path = os.path.abspath(data_path)
    logger.debug("Loading data file from %s", abs_path)

    if not os.path.exists(data_path):
        logger.error("Data file not found: %s", abs_path)
        raise FileNotFoundError(f"Data file not found: {abs_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.exception("Error occurred while reading CSV: %s", abs_path)
        raise

    logger.info("Data loaded successfully: shape=%s", df.shape)
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame by:
      - Dropping 'tweet_id' if present
      - Keeping only rows with sentiments in {'happiness', 'sadness'}
      - Encoding 'sentiment' to 1 (happiness) and 0 (sadness)

    Returns:
        pd.DataFrame: Processed DataFrame with binary-encoded 'sentiment'.
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Drop 'tweet_id' if present
    before_cols = set(df.columns)
    df = df.drop(columns=["tweet_id"], errors="ignore")
    after_cols = set(df.columns)
    dropped = before_cols - after_cols
    if dropped:
        logger.debug("Dropped columns: %s", list(dropped))

    # check required column
    if "sentiment" not in df.columns:
        logger.error("Required column 'sentiment' is missing.")
        raise KeyError("Required column 'sentiment' not found in the dataset.")

    # Filter only target classes
    valid = {"happiness", "sadness"}
    mask = df["sentiment"].isin(valid)
    kept = int(mask.sum())
    total = int(len(df))
    dropped_rows = total - kept
    if dropped_rows:
        logger.info("Filtered out %d rows with sentiments not in %s", dropped_rows, valid)

    final_df = df.loc[mask].copy()

    if final_df.empty:
        logger.error("No rows left after filtering sentiments %s", valid)
        raise ValueError("No rows left after filtering the required sentiments.")
    
    #Encode target
    mapping = {"happiness": 1, "sadness": 0}
    final_df.loc[:, "sentiment"] = final_df["sentiment"].map(mapping).astype("int64")

    logger.info("Processed data shape: %s; class balance -> happiness(1)=%d, sadness(0)=%d",
                final_df.shape,
                int((final_df["sentiment"] == 1).sum()),
                int((final_df["sentiment"] == 0).sum()))
    
    return final_df

def save_data(data_path: str,train_data: pd.DataFrame,test_data: pd.DataFrame, train_filename: str, test_filename: str) -> None:
    """
    Writes csv file to the specified path
    """
    
    if not isinstance(data_path, str) or not data_path.strip():
        raise ValueError("data_path must be a non-empty string.")
    if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
        raise ValueError("train_data and test_data must be DataFrames.")

    os.makedirs(data_path,exist_ok=True)

    train_csv = os.path.join(data_path, f"{train_filename}.csv")
    test_csv = os.path.join(data_path, f"{test_filename}.csv")

    train_data.to_csv(train_csv,index=False,encoding='utf-8')
    test_data.to_csv(test_csv,index=False,encoding='utf-8')

def main() -> None:

    try:
        # 1) Load and validate params
        params = load_params("params.yaml")
        test_size = params.get("data_ingestion", {}).get("test_size", None)

        if not isinstance(test_size, (int, float)):
            logger.error("Invalid type for data_ingestion.test_size: %r", test_size)
            raise ValueError("data_ingestion.test_size must be a float in (0,1)")
        test_size = float(test_size)
        if not (0 < test_size < 1):
            logger.error("data_ingestion.test_size out of range (0,1): %s", test_size)
            raise ValueError("data_ingestion.test_size must be in (0,1)")

        logger.info("Params loaded. test_size=%s", test_size)

        # 2) Load data
        df = load_data(os.path.join("data","external","tweet_emotions.csv"))

        # 3) Process data (drops tweet_id, filters classes, encodes target)
        final_df = process_data(df)

        # 4) Train/Test split with stratification to preserve class distribution
        logger.info("Splitting data")
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        # 5) Save data
        data_path = os.path.join("data", "raw")
        save_data(
            data_path=data_path,
            train_data=train_data,
            test_data=test_data,
            train_filename="train",
            test_filename="test"
        )

        logger.info("Data ingestion completed successfully.")


    except Exception:
        logger.exception("Pipeline failed in main()")
        raise  # re-raise so CLI/CI fails appropriately

if __name__ == "__main__":
    main()








