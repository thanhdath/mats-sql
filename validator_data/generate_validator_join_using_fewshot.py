import json
from tqdm import tqdm
from validator_data.validator import ValidatorJOIN, _execute_sql, _make_str_response, is_execution_correct, ValidatorJOINWithTrueSQL
import argparse
import os
import re
from multiprocessing import Pool

def process_sample(sample):
    """Process a single sample."""

    validator = Validator(endpoint_type=args.endpoint_type)

    try:
        true_execution_result = _execute_sql("./" + sample['db_path'], sample['sql'])

        sample['predict_sql'] = sample['predict_sqls'][0]
        prompt, answer, execution_result = validator.validate(sample)
        is_correct = is_execution_correct(true_execution_result[0], execution_result[0])

        sample['prompt_validator_join'] = prompt
        sample['is_correct'] = is_correct
        sample['feedback_conclude'] = answer is not None and 'Conclude: correct' in answer
        sample['validator_join'] = answer
        sample['true_result'] = _make_str_response(*true_execution_result)
        sample['pred_result'] = _make_str_response(*execution_result)

        return sample
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../temp/codes/eval_codes-1b.json')
    parser.add_argument('--output_file', type=str, default='bird_validator_join.jsonl')
    parser.add_argument('--endpoint_type', type=str, default='llamacpp', choices=['vllm', 'llamacpp', 'openai'])
    parser.add_argument('--use_hidden_sql', action='store_true')
    args = parser.parse_args()

    if args.use_hidden_sql:
        Validator = ValidatorJOINWithTrueSQL
    else:
        Validator = ValidatorJOIN

    data = []
    with open(args.input_file) as fp:
        for line in fp:
            data.append(json.loads(line))

    # Load saved output file if exists
    processed_keys = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                processed_keys.add((sample['db_id'], sample['question']))

    # Determine samples to process
    samples_to_process = [sample for sample in data if (sample['db_id'], sample['question']) not in processed_keys]

    # Open output file in append mode
    with open(args.output_file, 'a') as output_file:
        with Pool(8) as pool:
            for result in tqdm(pool.imap(process_sample, samples_to_process), total=len(samples_to_process)):
                if result is not None:
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
