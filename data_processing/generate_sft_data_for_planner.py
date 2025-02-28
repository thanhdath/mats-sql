import argparse
import os
import json
import re
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
from planner import _make_str_response, _execute_sql, is_execution_correct
from utils import norm_sql_query
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../data/multi-agents/planner/gpt-4o-mini-planner_combine_bird_with_evidence_train.jsonl')
parser.add_argument('--raw_train_file', type=str, default='../data/multi-agents/planner/gpt-4o-mini-planner_combine_bird_with_evidence_train.jsonl')
parser.add_argument('--output_dir', type=str, default='../data/multi-agents/planner/sft-gpt-4o-mini-planner_combine_bird_with_evidence_train/')
parser.add_argument('--error_file', type=str, default='../data/multi-agents/planner/gpt-4o-mini-planner_combine_bird_with_evidence_train-error-turn-1.jsonl')
parser.add_argument('--use_groundtruth', action='store_true')
parser.add_argument('--no_filter', action='store_true')
args = parser.parse_args()

PROMPT = """{schema}

Question: {question}
External knowledge: {evidence}

Planning:
"""
# PROMPT = """{schema}

# Question: {question}
# """

# Helper function for processing each sample
def process_sample(args):
    isample, sample, raw_sample, use_groundtruth, no_filter = args
    schema = raw_sample['schema_sequence']
    question = sample['question']
    evidence = sample['evidence']

    key = 'planner_combine_with_true_sql'
    feedback = sample[key]
    if feedback is None or len(feedback) == 0:
        return None, None  # Indicate empty result

    if isinstance(feedback, list):
        feedback = feedback[0]
 
    prompt = PROMPT.format(schema=schema, question=question, evidence=evidence)

    if use_groundtruth:
        completion = sample['sql']
        # completion = norm_sql_query(sample['sql'], raw_sample['schema'])
    else:
        # Extract SQL query using regex
        pred_sql_match = re.search(r"(?<=Final SQL query:).*?```(.*?)```", feedback, re.DOTALL)
        if pred_sql_match is None:
            pred_sql = " "
        else:
            pred_sql = pred_sql_match.group(1).strip()
            if pred_sql.startswith("sql"):
                pred_sql = pred_sql[3:].strip()

            # norm_pred_sql = norm_sql_query(pred_sql, raw_sample['schema'])
            # feedback = feedback.replace(pred_sql, norm_pred_sql)

        if not no_filter:
            true_result, has_error_true = _execute_sql("./" + sample["db_path"], sample["sql"])
            pred_result, has_error_pred = _execute_sql("./" + sample["db_path"], pred_sql)
            # norm_pred_result, has_error_pred = _execute_sql("./" + sample["db_path"], norm_pred_sql)

            # if not is_execution_correct(pred_result, norm_pred_result):
            #     # print to debug
            #     print("-" * 20)
            #     print("Norm SQL:", norm_pred_sql)
            #     print("Pred SQL:", pred_sql)
            #     print("Norm Result:", norm_pred_result)
            #     print("Pred Result:", pred_result)

            if not is_execution_correct(true_result, pred_result):
                # sample['true_result'] = _make_str_response(true_result, has_error_true)
                # sample['pred_result'] = _make_str_response(pred_result, has_error_pred)
                return None, sample  # Return sample with error

        completion = feedback if not isinstance(feedback, list) else feedback[0]
    prompt_id = f"{isample}"

    return {
        'prompt_id': prompt_id,
        'messages': {
            'prompt': prompt,
            'completion': completion
        }
    }, None  # Indicate valid result


if __name__ == "__main__":
    # Load data from input files
    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    raw_data = json.load(open(args.raw_train_file))

    # Prepare arguments for each sample to process
    samples_args = [(i, data[i], raw_data[i], args.use_groundtruth, args.no_filter) for i in range(len(data))]

    # Run parallel processing with 24 processes
    sft_data = []
    error_data = []
    with Pool(24) as pool:
        for result, error in tqdm(pool.imap_unordered(process_sample, samples_args), total=len(data)):
            if result:
                sft_data.append(result)
            if error:
                error_data.append(error)
    # for sample_arg in tqdm(samples_args):
    #     result, error = process_sample(sample_arg)
    #     if result:
    #         sft_data.append(result)
    #     if error:
    #         error_data.append(error)

    # Create datasets
    dataset = DatasetDict({
        'train': Dataset.from_list(sft_data),
        'test': Dataset.from_list(sft_data[:100]),
    })
    print(dataset)

    # Save the dataset
    dataset.save_to_disk(args.output_dir)

    # Write error data to JSONL file
    with open(args.error_file, 'w') as output_file:
        for sample in error_data:
            output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
