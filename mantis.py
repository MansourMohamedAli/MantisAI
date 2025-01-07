import pandas as pd
from pathlib import Path
from fuzzywuzzy import fuzz
from pydantic import BaseModel
from typing import Literal
from ollama import Client
import json

MODEL = "llama3.2"
# MODEL = "granite3-dense:8b"
# MODEL = "qwen2.5-coder:14b"
RESULT_THRESHOLD = 80

root = Path().resolve()
data_path = root / "data"
csv_file = data_path / "mantis_export.csv"

csv_df = pd.read_csv(csv_file)


def fuzzy_search(row, search):
    print(row)
    print(fuzz.ratio(row, search))
    # return fuzz.ratio(row, search)


search = "loca abort"
filt = csv_df['Description'].notna()
csv_df = csv_df[filt]
csv_df['match_%'] = csv_df['Description'].apply(lambda row:fuzzy_search(row, search))