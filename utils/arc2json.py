import pyarrow.parquet as pq
import pandas as pd
import json


def transform_data(df):
    result = []
    for index, row in df.iterrows():
        id_val = row["id"]
        question_stem = row["question"]
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        answer_key = row["answerKey"]
        
        choices = []
        for text, label in zip(choices_text, choices_label):
            choice_dict = {"text": text.strip(), "label": label.strip()}
            choices.append(choice_dict)
        
        question = {"stem": question_stem.strip(), "choices": choices}
        
        transformed_row = {"id": id_val.strip(), "question": question, "answerKey": answer_key.strip()}
        result.append(transformed_row)
    return result


def save_as_jsonl(data, filename):

    with open(filename, 'w') as f:
        for item in data:
            json_str = json.dumps(item, separators=(',', ':')) + '\n'
            f.write(json_str)


# read parquet file
parquet_file = pq.ParquetFile(r"C:\Users\12133\Downloads\test-00000-of-00001.parquet")
data = parquet_file.read().to_pandas()
df = pd.DataFrame(data)


transformed_data = transform_data(df)


save_as_jsonl(transformed_data, r'D:\ARC-Challenge-Test.jsonl')

