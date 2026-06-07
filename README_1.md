### Dataset: AG News Classification
- **Source:** [Kaggle - AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Original size:** 120,000 training rows, 7,600 test rows
- **Task:** Classify news articles into 4 categories — World, Sports, Business, Sci/Tech

### Why we subsampled to 50k rows
The full 120k dataset would eat up too much of Kaggle's free GPU time during fine-tuning. Since this project is about building the pipeline (not chasing accuracy), we grabbed 12,500 articles per class — enough to train a decent model without burning through our weekly 30-hour GPU quota.

## CSV files
We are going with csv instead of pkl because of:-
    - it is easy to load and inspect without code and safe.
    - text +label just 2 columns with 50k rows
    - not million rows or complex object and data types.
    - it is portable and see diff in PRs.

## tasks performed
| Step | Why we did it |
|------|---------------|
| Combined Title + Description into one `text` column | Gives the model more context per sample instead of two separate short strings |
| Fixed HTML artifacts (`#39;` → `'`, `quot;` → `"`, `&amp;` → `&`) | Raw data had encoding leftovers from web scraping that would confuse the tokenizer |
| Removed URLs (`http...`, `www...`) | Links don't carry meaning for topic classification — they're just noise |
| Collapsed extra whitespace | Multiple spaces/tabs mess with tokenization and add no value |
| Shifted labels from 1-4 to 0-3 | HuggingFace models expect zero-indexed labels; original dataset starts at 1 |
| Dropped empty/null rows | Can't train on nothing |
| Removed duplicate texts | Prevents data leakage and inflated metrics |
| Balanced sampling (equal per class) | Avoids the model being biased toward whichever class had more articles |

## NOT/preserved for transformers
- **No lowercasing** — DistilBERT's uncased tokenizer handles this internally, so doing it twice is pointless
- **No stopword removal** — transformer models actually use words like "the", "is", "at" for context understanding.             Transformers(BERT, DistilBERT or RoBERTa) care about grammar and sentence structure, whereas older models just cared about keywords.
- **No stemming/lemmatization** — like "better", "Best" to "good"--again, the tokenizer handles word pieces on its own; stemming would break it.The AI’s tokenizer relies on these specific word pieces (like ##ing, ##ed, or un##) to understand grammatical context, nuance, and meaning

### Output files
- `id2label.json` — maps model output indices to human-readable class names
- `data/train.csv` (50,000 rows) — (too large)
- `data/test.csv` (7,600 rows) — (smaller)

### Class distribution (after processing)
| Label | Class | Train samples | Test samples |
|-------|-------|---------------|--------------|
| 0 | World | 12,500 | 1,900 |
| 1 | Sports | 12,500 | 1,900 |
| 2 | Business | 12,500 | 1,900 |
| 3 | Sci/Tech | 12,500 | 1,900 |

### Using the Data Prep Script in Kaggle

To use the custom data processing script in a Kaggle notebook, you need to pull the repository into the Kaggle environment. 

The following code block will clone the repo if it's your first time running the cell, or refresh it with the latest changes if it has already been cloned:

```python
import os
import sys

repo_path = '/kaggle/working/mlops'

# 1. Clone the repo if it doesn't exist, otherwise pull the latest changes
if not os.path.exists(repo_path):
    !git clone [https://github.com/pratikktiwari/mlops.git](https://github.com/pratikktiwari/mlops.git)
else:
    print("Repository already exists. Pulling latest changes...")
    !cd {repo_path} && git pull

# 2. Add the repo root to sys.path so Python can find the script
if repo_path not in sys.path:
    sys.path.append(repo_path)

# 3. Import the cleaning function
from data_prep import process_split

# Load raw data from Kaggle and clean it using src/data_prep.py function
# ...
```
### Import the HF and WANDB keys ans check the evaluations.

