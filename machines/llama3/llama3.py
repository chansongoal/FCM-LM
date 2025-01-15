import os
import json
from datetime import datetime
import torch
from transformers import pipeline
# from ARC import ARCDataset
import numpy as np
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

class ARCChallengeSolver:
    def __init__(self, model_id, org_json_path, result_json_path, temp_id_file):
        self.model_id = model_id
        self.org_json_path = org_json_path
        self.result_json_path = result_json_path
        self.temp_id_file = temp_id_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_data(self):
        self.data = ARCDataset.load(self.org_json_path)
        print('org_data:', self.org_json_path, len(self.data))

    def prepare_inputs(self):
        input_columns = ['id', 'question'] + [f'text{i}' for i in range(1, 5)]
        self.answer_keys = []
        self.inputs = []
        self.ids = []
        for item in self.data:
            self.answer_keys.append(item['answerKey'])
            input_sequence_begin = f"<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\n"
            input_sequence_middle = f"Question: {item['question']}\n"
            options_content = ""
            for i, column in enumerate(input_columns[2:], start=0):
                if column in item:
                    option = 'ABCD'[i]
                    options_content = f"{option}: {item[column]}\n"
                input_sequence_middle += options_content
            input_sequence_end = "Your response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is"
            input_sequence = input_sequence_begin + input_sequence_middle + input_sequence_end
            self.inputs.append([input_sequence])
            self.ids.append(item['id'])

    def initialize_pipeline(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate_predictions(self):
        self.all_results = []

        correct_prompts = []
        correct_ids = []

        for i in range(len(self.inputs)):
            with open(self.temp_id_file, 'w') as file:
                file.write(f"{self.ids[i]}\n")
            prompt = self.inputs[i]
            result = self.pipeline(
                prompt,
                max_new_tokens=1,
                return_full_text=0,
                do_sample=False # guarantee the output is the same 
            )
            # print(f"prompt: {prompt}\n")
            # print(f"prediction: {result}\n")
            result_dict = {
                "prompt": prompt,
                "answer": result
            }
            self.all_results.append(result_dict)

        # # code below is for selecting the 100 samples with longest prompts (questions and answers)
        #     #gcs select the longest 100 prompts
        #     # Compare the result with the answer key
        #     if result[0][0]['generated_text'].strip() == self.answer_keys[i]:
        #         correct_prompts.append(self.inputs[i])
        #         correct_ids.append(self.ids[i])
    
        # # Sort the correct samples by the length of the result and select the top 100
        # lengths = np.array([len(s[0]) for s in correct_prompts])
        # print('max_min_mean_length: ', np.max(lengths), np.min(lengths), np.mean(lengths))
        # longest_100 = np.argsort(lengths)[-100:]    # longest idx in correct_prompts
        # longest100_ids = [correct_ids[i] for i in longest_100]  # longest id in the original data
        # print(longest100_ids)

        # filtered_data = []
        # org_file = "/home/gaocs/projects/FCM-LM/Data/llama3/csr/source/ARC-Challenge-Test.jsonl"
        # with open(org_file, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         entry = json.loads(line)
        #         if entry['id'] in longest100_ids:
        #             filtered_data.append(entry)

        # # Save the sampled results to a JSON file
        # with open("/home/gaocs/projects/FCM-LM/Data/llama3/csr/source/arc_challenge_sampled_longest100.jsonl", 'w', encoding='utf-8') as f:
        #     for entry in filtered_data:
        #         json.dump(entry, f, ensure_ascii=False)
        #         f.write('\n')


    def save_results(self):
        all_results_json = json.dumps(self.all_results, ensure_ascii=False, indent=4)
        with open(self.result_json_path, 'w', encoding='utf-8') as f:
            f.write(all_results_json)

    def calculate_accuracy(self):
        preds = [item['answer'][0][0]['generated_text'].strip() for item in self.all_results]
        accuracy = sum(pred == ans for pred, ans in zip(preds, self.answer_keys)) / len(self.answer_keys)
        return accuracy

    def llama_pipeline(self):
        self.load_data()
        self.prepare_inputs()
        self.initialize_pipeline()
        self.generate_predictions()
        # self.save_results()
        accuracy = self.calculate_accuracy()
        print(f"Accuracy: {accuracy*100:.4f}")

if __name__ == "__main__":
    model_path = "/home/gaocs/models/llama/Meta-Llama-3-8B-Instruct"
    org_json_path = "/home/gaocs/projects/FCM-LM/Data/llama3/csr/source/arc_challenge_sampled_longest100.jsonl"
    result_json_path = 'result.json'
    temp_id_file = "/home/gaocs/projects/FCM-LM/Data/llama3/csr/source/temp_id.txt"

    solver = ARCChallengeSolver(model_path, org_json_path, result_json_path, temp_id_file)
    solver.llama_pipeline()