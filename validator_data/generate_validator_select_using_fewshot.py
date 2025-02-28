import json
import os
from tqdm import tqdm
from validator_data.validator import ValidatorSelect, _execute_sql, _make_str_response, is_execution_correct
import argparse
import re

# add parse for input data file (train, dev) and output_file
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./data/evaluate/phi-3-planner_combine_bird_062024_with_evidence_train.jsonl')
parser.add_argument('--output_file', type=str, default='bird_validator_select.jsonl')
parser.add_argument('--endpoint_type', type=str, default='llamacpp', 
                    choices=['vllm', 'llamacpp', 'openai'])
args = parser.parse_args()

data = []
with open(args.input_file) as fp:
    for line in fp:
        data.append(json.loads(line))

# load saved output file if exists
if os.path.exists(args.output_file):
    old_output = []
    with open(args.output_file, 'r') as f:
        for line in f:
            old_output.append(json.loads(line))
    data[:len(old_output)] = old_output
else:
    old_output = []

# open jsonl file for append contents
output_file = open(args.output_file, 'a+')

validator = ValidatorSelect(endpoint_type=args.endpoint_type)

for isample in tqdm(range(len(old_output), len(data)), total=len(data)-len(old_output)):
    sample = data[isample]

    true_execution_result = _execute_sql("./" + sample['db_path'], sample['sql'])

    pred_sql_match = re.search(r"(?<=Final SQL query:).*", sample['planner'], re.DOTALL) 
    if pred_sql_match is None: continue

    pred_sql = pred_sql_match.group().replace("sql", "").replace("```", "").strip()
    sample['predict_sql'] = pred_sql


    answer, execution_result = validator.validate(sample)
    is_correct = is_execution_correct(true_execution_result[0], execution_result[0])

    print("-"*20)
    print("Is correct: ", is_correct)
    print(answer)

    sample['is_correct'] = is_correct
    sample['feedback_conclude'] = answer is not None and 'Conclude: correct' in answer
    sample['validator_select'] = answer

    sample['true_result'] = _make_str_response(*true_execution_result)
    sample['pred_result'] = _make_str_response(*execution_result)

    # json.dump(data[:isample+1], open(args.output_file, 'w+'), ensure_ascii=False, indent=4)
    # write new sample in jsonl file
    output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
    