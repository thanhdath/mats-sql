import json
from tqdm import tqdm
import functools
import sqlite3
import argparse
import re
import pandas as pd
from utils.db_utils import check_sql_executability
from data_processing.planner import get_answer_openai
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from validator_data.validator import _execute_sql
import traceback
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )
MODEL = "gpt-4o-mini"

# FOR llama
# client.base_url = "http://localhost:8000/v1"
# client.api_key = "no-key"
# MODEL = "Qwen-72B"

import sys
sys.path.append("test_suite_sql_eval/")
from exec_eval import eval_exec_match

def extract_sql_in_code_block(pred_sql_text):
    sql_block_match = re.search(r"```(.+?)```", pred_sql_text, re.DOTALL)

    if sql_block_match:
        sql_query = sql_block_match.group(1).strip()
        if sql_query is not None and sql_query.startswith("sql"):
            sql_query = sql_query.replace("sql", "")
        return sql_query
    else:
        return pred_sql_text


def get_executable_sql(db_path, sql_queries):
    for sql_query in sql_queries:
        if check_sql_executability(sql_query, db_path) is None:
            return sql_query
    return sql_queries[0] if len(sql_queries) > 0 else None

def filter_fixed_sql_different_than_pred_sql(fixed_sqls, pred_sqls):
    # norm \s+ to " " first
    fixed_sqls = [re.sub(r"\s+", " ", x).strip() for x in fixed_sqls]
    pred_sqls = [re.sub(r"\s+", " ", x).strip() for x in pred_sqls]

    return [x for x in fixed_sqls if x not in pred_sqls]

def process_sample(sample, dev_data):
    true_sql = sample['sql']
    all_sqls = []
    fixed_sqls = [x for x in sample.get('fixed_sqls', []) if x is not None]
    fixed_sqls = [extract_sql_in_code_block(x) for x in fixed_sqls]
    # fixed_sqls = filter_fixed_sql_different_than_pred_sql(fixed_sqls, sample['predict_sqls'])

    all_sqls.extend(fixed_sqls)
    all_sqls.extend(sample['predict_sqls'])
    # all_sqls.extend([''])
    if 'old_sqls' in sample:
        all_sqls.extend(sample['old_sqls'])

    pred_sql = get_executable_sql(sample["db_path"], all_sqls)
    if pred_sql is None:
        pred_sql = all_sqls[0]

    pred_sql = pred_sql.replace("\n", " ").strip()
    pred_result, _ = _execute_sql(sample["db_path"], pred_sql)
    true_result, has_error = _execute_sql(sample["db_path"], true_sql)

    try:
        if "spider" in dev_data:
            correct = eval_exec_match(sample['db_path'], pred_sql, sample['sql'], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False)
        else:
            correct = set(true_result) == set(pred_result)
    except Exception as err:
        print(err)
        correct = True

    if "spider" in dev_data:
        if len(fixed_sqls) > 0:
            fixed_sql = get_executable_sql(sample["db_path"], fixed_sqls)
            try:
                correct_after_fix = eval_exec_match(sample['db_path'], fixed_sql, sample['sql'], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False)
            except:
                correct_after_fix = True
        else:
            correct_after_fix = None

        sql_before_fix = get_executable_sql(sample["db_path"], sample['predict_sqls'])
        try:
            correct_before_fix = eval_exec_match(sample['db_path'], sql_before_fix, sample['sql'], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False)
        except:
            correct_before_fix = True
    else:
        pred_fix_sqls = [_execute_sql(sample["db_path"], pred_sql)[0] for pred_sql in fixed_sqls]
        correct_after_fix = any([set(true_result) == set(pred_result) for pred_result in pred_fix_sqls]) if pred_fix_sqls else None
        pred_results_before_fix = [_execute_sql(sample["db_path"], pred_sql)[0] for pred_sql in sample['predict_sqls']]
        correct_before_fix = any([set(true_result) == set(pred_result) for pred_result in pred_results_before_fix])

    if type(true_result) == list:
        true_result = true_result[:10]
    if type(pred_result) == list:
        pred_result = pred_result[:10]

    if 'feedback_selects' in sample:
        select_correct = sample['feedback_selects'][0] is None or 'Conclude: correct' in sample['feedback_selects'][0]
        condition_correct = sample['feedback_conditions'] [0]is None or 'Conclude: correct' in sample['feedback_conditions'][0]
        join_correct = sample['feedback_joins'][0] is None or 'Conclude: correct' in sample['feedback_joins'][0]
        order_correct = sample['feedback_orders'][0] is None or 'Conclude: correct' in sample['feedback_orders'][0]

        sample['select_correct'] = select_correct
        sample['condition_correct'] = condition_correct
        sample['join_correct'] = join_correct
        sample['order_correct'] = order_correct

    sample['correct_before_fix'] = correct_before_fix
    sample['correct_after_fix'] = correct_after_fix
    sample['true_result'] = str(true_result)
    sample['pred_result'] = str(pred_result)
    sample['is_correct'] = correct
    
    return sample

import multiprocessing
import json
import re

EOS_TOKEN = '<|eot_id|>'
ASSISTANT_TOKEN = '<|start_header_id|>assistant<|end_header_id|>'
USER_TOKEN = '<|start_header_id|>user<|end_header_id|>'
PROPMT_FIX = USER_TOKEN + """
{schema}

Question: {question}
External knowledge: {evidence}

Generated SQL query: {sql_query}

Execution response:
{execution_response}

Feedback for the SQL query:
{feedback_select}

{feedback_condition}

{feedback_join}

{feedback_order}

FIXED SQL:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN

def process_feedback_message_from_completion(prompt, answer, token):
    if prompt is None:
        prompt = ''
    
    if answer is None:
        return f"{token}\nNone"

    answer = prompt.split("Feedback:")[-1] + answer
    answer = answer.replace('<|assistant|>', '').replace('<|end|>', '').strip()
    answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>', '').replace('<|eot_id|>', '').strip()
    return answer

def modify_sample(sample):
    if not sample['is_correct']:
        # if sample['prompt_fix'][0] is None:
        #     return sample  # Return unmodified sample if there's no prompt_fix

        feedback_select = process_feedback_message_from_completion(sample['prompt_feedback_select'][0], sample['feedback_selects'][0], 'SELECT.')
        feedback_condition = process_feedback_message_from_completion(sample['prompt_feedback_condition'][0], sample['feedback_conditions'][0], 'CONDITION.')
        feedback_join = process_feedback_message_from_completion(sample['prompt_feedback_join'][0], sample['feedback_joins'][0], 'JOIN.')
        feedback_order = process_feedback_message_from_completion(sample['prompt_feedback_order'][0], sample['feedback_orders'][0], 'ORDER BY.')

        if sample['select_correct']:
            feedback_select = ""
        if sample['condition_correct']:
            feedback_condition = ""
        if sample['join_correct']: 
            feedback_join = ""
        if sample['order_correct']:
            feedback_order = ""

        if 'prompt_fix' not in sample:
            prompt_fix = PROPMT_FIX.format(
                schema=sample['schema_sequence'],
                question=sample['question'],
                evidence=sample['evidence'],
                sql_query=sample['predict_sqls'][0],
                execution_response=sample['pred_result'][0],
                feedback_select=feedback_select,
                feedback_condition=feedback_condition,
                feedback_join=feedback_join,
                feedback_order=feedback_order
            )
            text = """In your system, there are 3 agents.
- Planner agent who write a SQL query based on database schema and given question.
- Feedback agents who execute the SQL query and provide feedback to the planner agent. There are 4 types of feedback agents: SELECT, CONDITION, JOIN, and ORDER BY.
- Fix agent who corrects the SQL query based on feedback from the feedback agents.
Known that the generated sql is incorrect, and at least one of the feedbacks is incorrect. Read the correct SQL and the feedbacks from the feedback agents. Modify the feedbacks with  A SHORT REASON AND GUIDE to fix, so that the fix agent can correct the SQL query. If there is SQL syntax error, add comment on SELECT validator on how to fix it.
The modification must be slight differences from the original feedbacks. The feedbacks must be in the same format as the original feedbacks. The feedback must be end with "Conclude: correct" or "Conclude: incorrect", only conclude at the end of feedback. Only modify the feedbacks containing in the prompt_fix, for example if the prompt_fix contains feedbacks for SELECT and JOIN, only modify the feedbacks for SELECT and JOIN.

Example Feedback CONDITION. Follow this format to modify the feedback:
- The query uses:
   1. Condition in SELECT ```schools.school```. This selects the school names from the `schools` table.
   2. Condition in WHERE ```satscores.numge1500 > 500 AND schools.magnet = 1```. This filters for schools with more than 500 SAT test takers and that are magnet schools or offer a magnet program.

- Based on the question:
   1. 'schools with the SAT test takers of over 500': The query correctly filters for schools with SAT test takers greater than 500 using the condition ```satscores.numge1500 > 500```.
   2. 'magnet schools or offer a magnet program': The query correctly filters for magnet schools using the condition ```schools.magnet = 1```.

- However, the execution response shows that the result is an empty DataFrame, indicating that there are no records that meet the criteria specified in the WHERE clause. This could mean that there are no schools in the database that have both more than 500 SAT test takers and are classified as magnet schools.

The SQL query should checks for schools that are either classified as magnet schools or have a school type that includes "magnet" in its description (schools.magnet = 1 OR schools.soctype LIKE '%magnet%').

- Conclude: incorrect.


If there is no records, mainly because the SQL query is wrong, do not ask for verifying the data but determine the reason and the way to fix the SQL query. Some reasons that causes the SQL to return incorrect results:
- Use conditions on wrong columns (the columns don't contain the value used in the condition), leading to None or empty results.
- Not filter None values in the condition since some columns may contain None values, leading to None or empty results.
- JOIN unncessary tables leading to empty records after joining.
- Select more or less than the asked columns.

Answer in JSON format:
[{
"feedback_token": [one of the feedback tokens SELECT, JOIN, CONDITION, ORDER],
"feedback": [the modified feedback]
},]
Answer directly without any additional information.
"""
            text += f"Correct SQL: {sample['sql']}\n\n"
            text += f"The prompt to fix agent which contains feedbacks:\n{prompt_fix}\n"
        else:
            # prompt_fix = sample['prompt_fix'][0]
            prompt_fix = PROPMT_FIX.format(
                schema=sample['schema_sequence'],
                question=sample['question'],
                evidence=sample['evidence'],
                sql_query=sample['predict_sqls'][0],
                execution_response=sample['pred_result'][0],
                feedback_select=feedback_select,
                feedback_condition=feedback_condition,
                feedback_join=feedback_join,
                feedback_order=feedback_order
            )

            text = """In your system, there are 3 agents.
- Planner agent who write a SQL query based on database schema and given question.
- Feedback agents who execute the SQL query and provide feedback to the planner agent. There are 4 types of feedback agents: SELECT, CONDITION, JOIN, and ORDER BY.
- Fix agent who corrects the SQL query based on feedback from the feedback agents.
Known that the fix sql is incorrect. Read the correct SQL and the feedbacks from the feedback agents. Modify the feedbacks with A SHORT REASON AND GUIDE to fix, so that the fix agent can correct the SQL query.
The modification must be slight differences from the original feedbacks. The feedbacks must be in the same format as the original feedbacks. The feedback must be end with "Conclude: correct" or "Conclude: incorrect", only conclude at the end of feedback. Only modify the feedbacks containing in the prompt_fix, for example if the prompt_fix contains feedbacks for SELECT and JOIN, only modify the feedbacks for SELECT and JOIN.

Example Feedback CONDITION. Follow this format to modify the feedback:
- The query uses:
   1. Condition in SELECT ```schools.school```. This selects the school names from the `schools` table.
   2. Condition in WHERE ```satscores.numge1500 > 500 AND schools.magnet = 1```. This filters for schools with more than 500 SAT test takers and that are magnet schools or offer a magnet program.

- Based on the question:
   1. 'schools with the SAT test takers of over 500': The query correctly filters for schools with SAT test takers greater than 500 using the condition ```satscores.numge1500 > 500```.
   2. 'magnet schools or offer a magnet program': The query correctly filters for magnet schools using the condition ```schools.magnet = 1```.

- However, the execution response shows that the result is an empty DataFrame, indicating that there are no records that meet the criteria specified in the WHERE clause. This could mean that there are no schools in the database that have both more than 500 SAT test takers and are classified as magnet schools.

The SQL query should checks for schools that are either classified as magnet schools or have a school type that includes "magnet" in its description (schools.magnet = 1 OR schools.soctype LIKE '%magnet%').

- Conclude: incorrect.

If there is no records, mainly because the SQL query is wrong, do not ask for verifying the data but determine the reason and the way to fix the SQL query. Some reasons that causes the SQL to return incorrect results:
- Use conditions on wrong columns (the columns don't contain the value used in the condition), leading to None or empty results.
- Not filter None values in the condition since some columns may contain None values, leading to None or empty results.
- JOIN unncessary tables leading to empty records after joining.
- Select more or less than the asked columns.


Answer in JSON format:
[{
"feedback_token": [one of the feedback tokens SELECT, JOIN, CONDITION, ORDER],
"feedback": [the modified feedback]
},]
Answer directly without any additional information.
"""
            text += f"Correct SQL: {sample['sql']}\n\n"
            text += f"The prompt to fix agent which contains feedbacks:\n{prompt_fix}\n"
            text += f"\n\nThe fixed sql: {sample['fixed_sqls'][0]}\n"

        try:
            prompt = text

            answer = get_answer_openai(client, [{'role': 'user', 'content': prompt}], model=MODEL)[0]
            print(answer)

            # Extract JSON from ```json``` block
            completion = re.search(r"```(.+)```", answer, re.DOTALL)
            if completion is None:
                completion = answer
            else:
                completion = completion.group(1).strip()
            if completion.startswith("json"):
                completion = completion[4:]

            try:
                completions = json.loads(completion)
            except Exception as err:
                print(traceback.format_exc())
                print(f"Error JSON completion: {completion}")
                return sample

            for completion in completions:
                feedback_token = completion['feedback_token']
                feedback = completion['feedback']

                if feedback_token == 'SELECT':
                    sample['modified_feedback_selects'] = [feedback]
                elif feedback_token == 'CONDITION':
                    sample['modified_feedback_conditions'] = [feedback]
                elif feedback_token == 'JOIN':
                    sample['modified_feedback_joins'] = [feedback]
                elif feedback_token == 'ORDER BY':
                    sample['modified_feedback_orders'] = [feedback]
        except Exception as err:
            print(traceback.format_exc())
            print(f"Error processing sample: {sample['db_id']} {sample['question']}")

    return sample  # Return the modified sample

def load_previous_results(progress_file):
    if os.path.exists(progress_file):
        print(f"Loading previous progress from {progress_file}...")
        with open(progress_file, 'r') as f:
            return {sample['db_id'] + " " + sample['question']: sample for sample in map(json.loads, f)}
    return {}

def save_progress_to_file(processed_samples, progress_file):
    with open(progress_file, 'a') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample) + '\n')

def process_samples_in_parallel(samples, progress_file):
    # Load previously saved results
    processed_keys = set(load_previous_results(progress_file).keys())
    samples_to_process = [sample for sample in samples if sample['db_id'] + " " + sample['question'] not in processed_keys]

    with multiprocessing.Pool(8) as pool:
        # Wrap the imap function with tqdm to show a progress bar
        for sample in tqdm(pool.imap(modify_sample, samples_to_process), total=len(samples_to_process), desc="Processing Samples"):
            save_progress_to_file([sample], progress_file)  # Save each processed sample
    
    #for sample in samples_to_process:
    #    sample = modify_sample(sample)
    #    save_progress_to_file([sample], progress_file)  # Save each processed sample
        
    print(f"Progress saved to {progress_file}.")
    return progress_file  # Returning the file for reference

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_sample_with_index(args):
    """Helper function to process a sample with its index."""
    index, sample, dev_file = args
    processed_sample = process_sample(sample, dev_data=dev_file)
    return index, processed_sample

def process_samples_in_order(samples, dev_file):
    """Process samples in parallel while maintaining order."""
    args_list = [(index, sample, dev_file) for index, sample in enumerate(samples)]
    results = [None] * len(samples)  # Preallocate list to maintain order

    with ProcessPoolExecutor(max_workers=24) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sample_with_index, arg): arg[0] for arg in args_list}

        # Process completed tasks
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples"):
            index, processed_sample = future.result()
            results[index] = processed_sample

    return results



def main():
    parser = argparse.ArgumentParser(description='Process SQL evaluation for datasets.')
    parser.add_argument('--pred_file', default='data/llm_alignment/spider-p1_llama-3-end2end-spider_train_fix.jsonl', type=str)
    args = parser.parse_args()

    progress_file = args.pred_file.replace('.jsonl', '_progress.jsonl')

    if "spider_dev" in args.pred_file:
        args.dev_file = "data/sft_data_collections/spider/dev.json"
    elif "spider_dk" in args.pred_file:
        args.dev_file = 'data/sft_spider_dk_text2sql.json'
    elif "spider_realistic" in args.pred_file:
        args.dev_file = 'data/sft_spider_realistic_text2sql.json'
    elif "spider_syn" in args.pred_file:
        args.dev_file = 'data/sft_spider_syn_text2sql.json'
    elif "bird" in args.pred_file and "dev" in args.pred_file:
        args.dev_file = 'data/full_value_matching_schema_insight_bird_062024_with_evidence_dev_text2sql.json'
    elif "bird" in args.pred_file and "train" in args.pred_file:
        args.dev_file = 'data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json'
    elif "spider_train" in args.pred_file:
        args.dev_file = "data/sft_data_collections/spider/train.json"
    
    if 'spider' in args.pred_file:
        dataname = 'spider'
    elif 'bird' in args.pred_file:
        dataname = 'bird'
    else:
        raise Exception("Unhandled data")

    results_dict = {}
    with open(args.pred_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            results_dict[f"{sample['db_id']} {sample['question']}"] = sample

    with open(args.dev_file) as dev_fp:
        dev_data = json.load(dev_fp)

    dev_keys = [f"{sample['db_id']} {sample['question']}" for sample in dev_data]
    results = [results_dict[key] for key in dev_keys if key in results_dict]

    # Replace the old loop with the new function
    if os.path.isfile('processed_results.json'):
        with open('processed_results.json', 'r') as f:
            processed_results = json.load(f)
    else:
        processed_results = process_samples_in_order(results, dev_file=args.dev_file)
        with open('processed_results.json', 'w') as f:
            f.write(json.dumps(processed_results))
    

    n_correct = sum(1 for sample in processed_results if sample['is_correct'])
    print('Acc before fix:', sum(x.get('correct_before_fix', 0) or 0 for x in processed_results) / len(processed_results))
    print('Acc after fix:', n_correct / len(processed_results))

    for sample in processed_results:
        for field in ['schema', 'table_labels', 'column_labels']:
            sample.pop(field, None)

    # Process samples in parallel and save progress incrementally
    processed_progress_file = process_samples_in_parallel(processed_results, progress_file)

    # Merge all saved progress into final output
    with open(processed_progress_file, 'r') as f:
        final_results = [json.loads(line) for line in f]

    # Dump the results to a JSON file, not JSONL
    output_file = args.pred_file.replace('.jsonl', '_modify_feedback.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to {output_file}.")

if __name__ == '__main__':
    main()
