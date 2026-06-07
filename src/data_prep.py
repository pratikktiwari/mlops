import os
import re
import json
import pandas as pd
import html

try:
    import kagglehub
except ImportError as e:
    raise ImportError(
        "The 'kagglehub' module was not found. "
        "If you are running this outside Kaggle, "
        "please install: pip install kagglehub"
    ) from e

SAMPLES_PER_CLASS_TRAIN = 12500
SAMPLES_PER_CLASS_TEST = 1900
RANDOM_SEED = 42

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # 1. Safely and completely unescape ALL HTML entities
    text = html.unescape(text)
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def process_split(df, samples_per_class=None):
    df = df.copy()
    
    # Wrapped to fix line length > 79
    df["text"] = (
        df["Title"].apply(clean_text) + 
        " " + 
        df["Description"].apply(clean_text)
    )
    
    df["label"] = df["Class Index"] - 1
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    
    if samples_per_class:
        df = df.groupby("label").apply(
            # Wrapped lambda to fix line length > 79
            lambda x: x.sample(
                n=min(samples_per_class, len(x)), 
                random_state=RANDOM_SEED
            )
        ).reset_index(drop=True)
        
    return df[["text", "label"]]

def main():
    # Wrapped download path to fix line length > 79
    path = kagglehub.dataset_download(
        "amananandrai/ag-news-classification-dataset"
    )
    train_raw = pd.read_csv(os.path.join(path, "train.csv"))
    test_raw = pd.read_csv(os.path.join(path, "test.csv"))
