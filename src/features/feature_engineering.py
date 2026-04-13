import numpy as np
import pandas as pd
import os
from typing import Optional,Any, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from src.data.data_ingestion import load_params,load_data,save_data,get_logger

logger = get_logger(__name__)

def fill_na_text(df: pd.DataFrame, text_col: str = "content", replace_with: str = "") -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Required text column '{text_col}' not found.")
    df = df.copy()
    df[text_col] = df[text_col].fillna(replace_with).astype(str)
    return df

def seperate_col(df: pd.DataFrame,col_name: str) -> np.ndarray:
    if col_name not in df.columns:
        raise KeyError(f"Required column '{col_name}' not found.")
    return df[col_name].values

def _validate_max_features(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError("feature_engineering.max_features must be None or a positive int (not bool).")
    if isinstance(val, (int, np.integer)):
        if val <= 0:
            raise ValueError("feature_engineering.max_features must be > 0.")
        return int(val)
    raise ValueError("feature_engineering.max_features must be None or a positive int.")

def main():
    try:
        params: Dict[str, Any] = load_params('params.yaml')
        fe_params = params.get("feature_engineering",{})
        max_features = _validate_max_features(fe_params.get("max_features"))

        logger.info(
            "Feature params -> max_features=%s",max_features
        )

        # 2) Load data
        train_data = load_data(os.path.join("data", "interim", "train_processed.csv"))
        test_data  = load_data(os.path.join("data", "interim", "test_processed.csv"))
        logger.info("Loaded interim data: train=%s, test=%s", train_data.shape, test_data.shape)

        # 3) Fill Na only for text column
        train_data = fill_na_text(train_data, text_col="content", replace_with="")
        test_data  = fill_na_text(test_data,  text_col="content", replace_with="")

        # 4) Separate X and y (keep target name consistent as 'sentiment')
        X_train = seperate_col(train_data, "content")
        y_train = seperate_col(train_data, "sentiment")

        X_test = seperate_col(test_data, "content")
        y_test = seperate_col(test_data, "sentiment")

        # Apply Tfidf (TfidfVectorizer)
        vectorizer = TfidfVectorizer(max_features=max_features)

        logger.info("Fitting TfidfVectorizer on train text...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.info(
            "Vectorized train: shape=%s, vocab_size=%d",
            X_train_tfidf.shape, len(vectorizer.vocabulary_)
        )

        logger.info("Transforming test text with fitted vectorizer...")
        X_test_tfidf = vectorizer.transform(X_test)
        logger.info("Vectorized test: shape=%s", X_test_tfidf.shape)

        # Convert Tfidf representations into DataFrames
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        # Store the features in data/processed
        data_path = os.path.join("data", "processed")

        save_data(data_path,train_df,test_df,'train_tfidf','test_tfidf')
        logger.info("Saved Tfidf features to %s", os.path.abspath(data_path))

        logger.info("Feature engineering completed successfully.")

    except Exception:
        logger.exception("Feature engineering pipeline failed.")
        raise
        
if __name__ == "__main__":
    main()