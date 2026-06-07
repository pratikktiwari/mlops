import os
import re
import json
import pandas as pd
try:
    import kagglehub
except ImportError as e:
    raise ImportError(
        "The 'kagglehub' module was not found. "
        "If you are running this script outside of Kaggle, please install it by running: pip install kagglehub"
    ) from e

SAMPLES_PER_CLASS_TRAIN = 12500
SAMPLES_PER_CLASS_TEST = 1900
RANDOM_SEED = 42


import html

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # 1. Safely and completely unescape ALL HTML entities
    # text = re.sub(r"#39;", "'", text)
    # text = re.sub(r"&amp;", "&", text)
    # text = re.sub(r'quot;', '"', text)
    text = html.unescape(text)
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def process_split(df, samples_per_class=None):
    df = df.copy()
    df["text"] = df["Title"].apply(clean_text) + " " + df["Description"].apply(clean_text)
    df["label"] = df["Class Index"] - 1
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    if samples_per_class:
        df = df.groupby("label").apply(
            lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=RANDOM_SEED)
        ).reset_index(drop=True)
    return df[["text", "label"]]


def main():
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    train_raw = pd.read_csv(os.path.join(path, "train.csv"))
    test_raw = pd.read_csv(os.path.join(path, "test.csv"))

    train_df = process_split(train_raw, SAMPLES_PER_CLASS_TRAIN)
    test_df = process_split(test_raw, SAMPLES_PER_CLASS_TEST)

    id2label = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
