import json
import os.path as osp

from datasets import Dataset

class ARCDataset:
    @staticmethod
    def load(path: str):
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for line in in_f:
                try:
                    item = json.loads(line.strip())
                    id = item['id']
                    question = item['question']
                    choices = question['choices']
                    num_choices = len(choices)

                    labels = [c['label'] for c in choices]
                    if item['answerKey'] not in labels:
                        raise ValueError(f"answerKey '{item['answerKey']}' not in labels {labels}")
                    answerKey_index = labels.index(item['answerKey'])
                    answerKey = 'ABCDE'[answerKey_index]

                    texts = [c['text'] for c in choices]

                    row = {
                        'id':id,
                        'question': question['stem'],
                        'answerKey': answerKey,
                    }
                    for i, text in enumerate(texts):
                        row[f'text{i+1}'] = text

                    rows.append(row)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
                except KeyError as e:
                    print(f"Missing key in data: {e}")
                except ValueError as e:
                    print(e)

            return rows



