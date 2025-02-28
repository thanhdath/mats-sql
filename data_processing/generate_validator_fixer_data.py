import argparse
import os
import json
from datasets import Dataset, DatasetDict
from planner import get_answer_llamacpp, get_answer_vllm, get_answer_openai
from openai import OpenAI
from dotenv import load_dotenv
from planner import _make_str_response, _execute_sql, is_execution_correct
import re
from utils import norm_sql_query
from tqdm import tqdm
from multiprocessing import Pool

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./data/multi-agents/fixed/gpt-4o-mini-validator-fixer-bird_with_evidence_train.jsonl')
parser.add_argument('--output_dir', type=str, default='./data/multi-agents/fixed/sft-gpt-4o-mini-validator-fixer-bird_with_evidence_train')
parser.add_argument('--num_processes', type=int, default=16)
args = parser.parse_args()

# Define the prompt template
PROMPT = """{schema}

Question: {question}
External knowledge: {evidence}

Generated SQL query: {sql_query}

Execution response:
{execution_response}

Feedback for the SQL query:
"""

COMPLETION = """<select>
{feedback_select}
</select>

<condition>
{feedback_condition}
</condition>

FIXED SQL: {fixed_sql}"""

def norm_feedback(feedback, token):
    feedback = token + feedback.split(token)[-1]
    return feedback

def extract_sql_in_code_block(pred_sql_text):
    sql_block_match = re.search(r"```(.+?)```", pred_sql_text, re.DOTALL)
    if sql_block_match:
        sql_query = sql_block_match.group(1).strip()
        if sql_query.startswith("sql"):
            sql_query = sql_query.replace("sql", "").strip()
        return sql_query
    else:
        return pred_sql_text

def process_sample(index_sample):
    index, sample = index_sample
    feedback_select = sample['validator_select'] or 'SELECT.\nNone'
    feedback_condition = sample['validator_condition'] or "CONDITION.\nNone"
    feedback_join = sample['validator_join'] or "JOIN.\nNone"
    feedback_join = "JOIN." + feedback_join.split("JOIN.")[-1]

    feedback_select = norm_feedback(feedback_select, "SELECT.")
    feedback_condition = norm_feedback(feedback_condition, "CONDITION.")
    feedback_join = norm_feedback(feedback_join, "JOIN.")

    prompt = PROMPT.format(
        schema=sample['schema_sequence'],
        question=sample['question'],
        evidence=sample['evidence'],
        sql_query=sample['predict_sql'],
        execution_response=sample['pred_result']
    )

    fixed_sql = sample['fixed_sql']
    if type(fixed_sql) == list:
        fixed_sql = fixed_sql[0]

    fixed_sql = extract_sql_in_code_block(fixed_sql)

    if fixed_sql != "None":
        true_result, has_error = _execute_sql("./" + sample["db_path"], sample["sql"])
        pred_result, has_error = _execute_sql("./" + sample["db_path"], fixed_sql)

        if not is_execution_correct(true_result, pred_result):
            print("-"*20)
            print('True:', true_result)
            print('Pred:', pred_result)
            # completion = norm_sql_query(sample['sql'], sample['schema'])
            fixed_sql = sample['sql']

    completion = COMPLETION.format(
        feedback_select=feedback_select,
        feedback_condition=feedback_condition,
        # feedback_join=feedback_join,
        fixed_sql=fixed_sql
    )

    return {
        'prompt_id': str(index),
        'messages': {
            'prompt': prompt,
            'completion': completion
        }
    }

def main():
    with open(args.input_file) as fp:
        data = [json.loads(line) for line in fp]

    with Pool(processes=args.num_processes) as pool:
        results = list(tqdm(pool.imap(process_sample, enumerate(data)), total=len(data)))

    sft_data = [result for result in results if result is not None]

    dataset = DatasetDict({
        'train': Dataset.from_list(sft_data),
        'test': Dataset.from_list(sft_data[:100]),
    })
    dataset.save_to_disk(args.output_dir)
    print(dataset)

if __name__ == "__main__":
    main()
