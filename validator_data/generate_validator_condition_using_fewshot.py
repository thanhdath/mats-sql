import json
from tqdm import tqdm
from validator import ValidatorCondition, _execute_sql, _make_str_response, is_execution_correct, ValidatorConditionWithTrueSQL
import argparse
import re
import os
from multiprocessing import Pool, Manager

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../temp/codes/eval_codes-1b.json')
parser.add_argument('--output_file', type=str, default='bird_validator_join.jsonl')
parser.add_argument('--endpoint_type', type=str, default='llamacpp', choices=['vllm', 'llamacpp', 'openai'])
parser.add_argument('--use_hidden_sql', action='store_true')
args = parser.parse_args()

def process_sample(args):
    idx, sample, endpoint_type, output_file_lock, output_file_path = args

    try:
        validator = Validator(endpoint_type=endpoint_type)

        true_execution_result = _execute_sql("./" + sample['db_path'], sample['sql'])

        pred_sql_match = re.search(r"(?<=Final SQL query:).*", sample['planners'][0], re.DOTALL)
        if pred_sql_match is None:
            return None

        pred_sql = pred_sql_match.group().replace("sql", "").replace("```", "").strip()
        sample['predict_sql'] = pred_sql

        prompt, answer, execution_result = validator.validate(sample)
        answer = answer[0]
        is_correct = is_execution_correct(true_execution_result[0], execution_result[0])

        sample['is_correct'] = is_correct
        sample['feedback_conclude'] = answer is not None and 'Conclude: correct' in answer
        sample['validator_condition'] = answer
        sample['true_result'] = _make_str_response(*true_execution_result)
        sample['pred_result'] = _make_str_response(*execution_result)

        # Write the result to the file
        with output_file_lock:
            with open(output_file_path, 'a') as output_file:
                output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

        return idx

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None

if __name__ == "__main__":
    if args.use_hidden_sql:
        Validator = ValidatorConditionWithTrueSQL
    else:
        Validator = ValidatorCondition

    # Load data
    data = []
    with open(args.input_file) as fp:
        for line in fp:
            data.append(json.loads(line))

    # Load old results
    if os.path.exists(args.output_file):
        processed_indices = set()
        with open(args.output_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                processed_indices.add(f"{result['db_id']} {result['question']}")
        print(f"Loaded {len(processed_indices)} previously processed samples.")
    else:
        processed_indices = set()

    # Filter data to process only unprocessed samples
    unprocessed_data = [
        (i, sample, args.endpoint_type, None, args.output_file)
        for i, sample in enumerate(data)
        if f"{sample['db_id']} {sample['question']}" not in processed_indices
    ]

    # Set up multiprocessing
    with Manager() as manager:
        output_file_lock = manager.Lock()

        # Add output_file_lock to each task
        tasks = [
            (idx, sample, args.endpoint_type, output_file_lock, args.output_file)
            for idx, sample, _, _, _ in unprocessed_data
        ]

        with Pool(processes=12) as pool:
            # Use tqdm to monitor progress
            for _ in tqdm(pool.imap_unordered(process_sample, tasks), total=len(tasks)):
                pass

    print("Processing completed.")
