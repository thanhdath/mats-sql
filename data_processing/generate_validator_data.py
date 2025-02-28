import json
import sqlite3
import multiprocessing.pool
import functools
from tqdm import tqdm
import pandas as pd
from validator import ValidatorJOIN, _execute_sql, _make_str_response, is_execution_correct
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../temp/codes/eval_codes-1b.json')
parser.add_argument('--output_file', type=str, default='bird_validator_join.jsonl')
parser.add_argument('--endpoint_type', type=str, default='llamacpp', choices=['vllm', 'llamacpp', 'openai'])
args = parser.parse_args()

data = json.load(open(args.input_file))

if os.path.exists(args.output_file):
    old_output = json.load(open(args.output_file))
    data[:len(old_output)] = old_output
else:
    old_output = []

# open jsonl file for append contents
output_file = open(args.output_file, 'a+')

validator = ValidatorJOIN(endpoint_type=args.endpoint_type)

for isample in tqdm(range(0, len(data)), total=len(data)):
    sample = data[isample]

    true_execution_result = _execute_sql("../" + sample['db_path'], sample['sql'])
    
    sql = sample['predict_sql']

    answer, execution_result = validator.validate(sample)
    is_correct = is_execution_correct(true_execution_result[0], execution_result[0])

    print("-"*20)
    print("Is correct: ", is_correct)
    print(answer)

    sample['is_correct'] = is_correct
    sample['feedback_conclude'] = answer is not None and 'Conclude: correct' in answer
    sample['validator_join'] = answer

    sample['true_result'] = _make_str_response(*true_execution_result)
    sample['pred_result'] = _make_str_response(*execution_result)

    del sample['table_labels']
    del sample['column_labels']
    del sample['schema']
    del sample['matched_contents']

    # json.dump(data[:isample+1], open(args.output_file, 'w+'), ensure_ascii=False, indent=4)
    # write new sample in jsonl file
    output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
    