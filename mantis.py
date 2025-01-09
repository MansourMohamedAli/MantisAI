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
RESULT_THRESHOLD = 60

class MantisResponse(BaseModel):
  solution: str
  reason: str

root = Path().resolve()
data_path = root / "data"
csv_file = data_path / "mantis_export.csv"

csv_df = pd.read_csv(csv_file)

def apply_ratio(row, search):
    return fuzz.ratio(row, search)
def apply_partial_ratio(row, search):
    return fuzz.partial_ratio(row, search)
def apply_token_sort_ratio(row, search):
    return fuzz.token_sort_ratio(row, search)
def apply_token_set_ratio(row, search):
    return fuzz.token_set_ratio(row, search)
def apply_weighted_ratio(row, search):
    return fuzz.WRatio(row, search)

filt = csv_df['Description'].notna()
csv_df = csv_df[filt]

search = "How to set up ThunderView Picture in Picture(PIP)?"
csv_df['ratio_%'] = csv_df['Summary'].apply(lambda row:apply_ratio(row, search))
# csv_df['partial_ratio_%'] = csv_df['Summary'].apply(lambda row:apply_partial_ratio(row, search))
# csv_df['token_sort_ratio_%'] = csv_df['Summary'].apply(lambda row:apply_token_sort_ratio(row, search))
# csv_df['token_set_ratio_%'] = csv_df['Summary'].apply(lambda row:apply_token_set_ratio(row, search))
# csv_df['weighted_ratio_%'] = csv_df['Description'].apply(lambda row:apply_weighted_ratio(row, search))

csv_df.sort_values(by='ratio_%', inplace=True, ascending=False)
csv_df = csv_df.head(2)
csv_df

def generate_response(search, fuzzy_match_list):
    client = Client()
    messages = [{'role': 'user',
                 'content': f'For user questions: {search}, search these bug reports that were ranked by match precentage: {fuzzy_match_list}. The match percentage \
for each bug report is enclosed by brackets, Like the following: [[match %]]. weigh the match % extremely heavily when picking a category.\
Remember, variables in the summary that end in parenthese such as thk_flownw(408) are not generic and most likely do not match the numbering scheme the \
user is asking about.'}]
    
    json_str = client.chat(model=MODEL, 
               messages=messages,
            #    options={'num_ctx': 16384, 'temperature': 0},
               options={'temperature': 0},
               format=MantisResponse.model_json_schema())['message']['content']
    response = json.loads(json_str)
    print('*' * 200)
    print(f'Response for "{search}": [{response['solution']}]\n')
    print(f'Reason for response: {response['reason']}')
    print(f'\nMATCH LIST:\n{fuzzy_match_list}')
    print('*' * 200)


row_dict = csv_df.to_dict(orient='records') 
results = ""
result_found = False
results_list = list()
for row in row_dict:
    # if int(row['ratio_%']) > RESULT_THRESHOLD:
    results += f'{row['Summary']}:\n \
{row['Description']}:\n \
{row['Notes']}: [[{row['ratio_%']}%]].\n \
end of DR {row['Id']}\n'
    result_found = True
    results_list.append(f'DR {row["Id"]}: {row['Summary']} ({row['Project']})')
        
# if not result_found:
#     results = f'No results over [[{RESULT_THRESHOLD}%]].'
# else:
#     generate_response(search, results)

generate_response(search, results)

for r in results_list:
    print(r)