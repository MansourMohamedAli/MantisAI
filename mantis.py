import pandas as pd
from pathlib import Path


def get_mantis_csv(export_path):
    root = Path().resolve()
    data_path = root / "data"
    csv_file = data_path / "mantis_export_all.csv"

    csv_df = pd.read_csv(csv_file)
    filt = (csv_df['Description'].notna()) & (csv_df['Status'] == 'Closed')
    csv_df = csv_df[filt]
    csv_df.set_index('Id', inplace=True)

    csv_to_llm_df = csv_df[['Summary', 'Description', 'Notes']]
    csv_to_llm_df = csv_to_llm_df.head(5)
    return csv_to_llm_df.to_csv(export_path,header=False)
    # return csv_to_llm_df.to_csv(export_path)


if __name__ == "__main__":
    get_mantis_csv('data/mantis.csv')