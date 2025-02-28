import json
import os
from tqdm import tqdm
from data_processing.planner import Planner, FixAgent, SelectionAgent, SelectionAgentWithSchema
import argparse
from multiprocessing import Pool
import requests
import re
from utils.db_utils import check_sql_executability
from validator_data.validator import ValidatorSelect, ValidatorJOIN, ValidatorOrder, ValidatorCondition, _make_str_response, _execute_sql
from copy import deepcopy
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np

def extract_sql_in_code_block(pred_sql_text):
    sql_block_match = re.search(r"```(.+?)```", pred_sql_text, re.DOTALL)

    if sql_block_match:
        sql_query = sql_block_match.group(1).strip()
        if sql_query.startswith("sql"):
            sql_query = sql_query.replace("sql", "")
        return sql_query
    else:
        return pred_sql_text


class PostProcessing:
    @staticmethod
    def post_process_sql(sql, schema):
        table_names = [table['table_name'] for table in schema['schema_items']]
        # replace this pattern table_name.table_name.column_name with table_name.column_name
        for table_name in table_names:
            sql = sql.replace(f"{table_name}.{table_name}.", f"{table_name}.")

        # sql = sql.lower()
        sql = re.sub("\s+", " ", sql)
        return sql

class MultiAgentSystem():
    def __init__(self, get_answer_func):
        self.planner = Planner(prompt_file='data_processing/prompts/zero_shot_prompt_planner.txt', 
                    endpoint_type='vllm')
        if args.model_name != 'codes':
#             self.planner.prompt_template = """{schema}

# Question: {question}
# External knowledge: {evidence}

# Planning:
# <|reserved_special_token_247|>"""

            self.planner.prompt_template =  USER_TOKEN + """
{schema}

Question: {question}
External knowledge: {evidence}

Planning:
""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN
            
        if args.model_name == 'nl2sql':
            self.planner.prompt_template =  """{schema}

Question: {question}{evidence}
<|eot_id|>"""

        self.validator_select = ValidatorSelect(endpoint_type='vllm')
        self.validator_condition = ValidatorCondition(endpoint_type='vllm')
        self.validator_join = ValidatorJOIN(endpoint_type='vllm')
        self.validator_order = ValidatorOrder(endpoint_type='vllm')

        self.validator_select.prompt_template = USER_TOKEN + """
Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Feedback:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN + """
SELECT.
1. Based on the SQL query, the query selects: {select_columns}"""

        '''
        self.validator_condition.prompt_template = USER_TOKEN + """Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Write feedback, include Conclude (incorrect or correct) at the end of your answer.
If there is a syntax error, write "Conclude: incorrect", then write the reason and guide to fix it.
Some error and how to fix:
- no such column, guide to add need tables in the JOIN.
- no such table, need write a correct table name.""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN + """
CONDITION."""'''
        
        self.validator_condition.prompt_template = USER_TOKEN + """
Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Feedback:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN + """
CONDITION.
"""

        self.validator_join.prompt_template = USER_TOKEN +  """
Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Feedback:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN + """
JOIN.
- The SQL query uses tables {used_tables}, joining them on foreign keys {used_fks}."""

        self.validator_order.prompt_no_none = USER_TOKEN + """
Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Feedback:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN + """
ORDER BY.
- The SQL query uses ```{order_by_clause}```.
- Based on the question, the query should use"""

        self.validator_order.prompt_has_none = USER_TOKEN + """
Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Feedback:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN + """
ORDER BY.
- The SQL query uses ```{order_by_clause}```.
- However, the column ```{order_by_column}``` has None values, so the SQL query need to add condition ```{order_by_column} IS NOT NULL``` to filter out None values.
- Conclude: incorrect."""

        self.fixed_sql_agent = FixAgent(USER_TOKEN + """
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

FIXED SQL:""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN
)
        
        self.selection_agent = SelectionAgent(endpoint_type='vllm')

        self.planner.get_answer = get_answer_planner
        self.validator_select.get_answer = lambda x: get_answer_validator('validator-select', x)
        self.validator_condition.get_answer = lambda x: get_answer_validator('validator-condition', x)
        self.validator_join.get_answer = lambda x: get_answer_validator('validator-join', x)
        self.validator_order.get_answer = lambda x: get_answer_validator('validator-order', x)
        self.fixed_sql_agent.get_answer = get_answer_fixed
        self.selection_agent.get_answer = get_answer_selection

    def _extract_sql_in_plan(self, plan):
        pred_sql_match = re.search(r'Final SQL query:\s*```(.*?)```', plan, re.DOTALL)
        if pred_sql_match is None:
            if plan.strip().startswith('SELECT'):
                pred_sql = plan.strip()
            else:
                # find ``` ``` block
                sql_block_match = re.search(r"```(.+?)```", plan, re.DOTALL)
                if sql_block_match:
                    pred_sql = sql_block_match.group(1).strip()
                else:
                    return None
        else:
            pred_sql = pred_sql_match.group(1).replace("sql", "").replace("```", "").strip()

        return pred_sql
    
    def generate_plans(self, sample):
        prompt_planner, plans = self.planner.generate(sample)

        plan_with_sqls = []
        added_sqls = set()

        for plan in plans:
            pred_sql = self._extract_sql_in_plan(plan)
            if pred_sql is None:
                continue
            pred_sql = PostProcessing.post_process_sql(pred_sql, sample['schema'])
            #print(f"pred_sql: {pred_sql}")
            #print(f"new pred_sql: {pred_sql}")
            if pred_sql not in added_sqls:
                added_sqls.add(pred_sql)
                plan_with_sqls.append((plan, pred_sql))

        good_plans = []
        good_plan_sqls = []
        
        for plan, pred_sql in plan_with_sqls:
            # if args.mode == 'test':
            #     execution_error = check_sql_executability(pred_sql, sample["db_path"])
            #     if execution_error is not None:
            #         continue
                
            #     good_plans.append(plan)
            #     good_plan_sqls.append(pred_sql)
            #     break
            # else:
                good_plans.append(plan)
                good_plan_sqls.append(pred_sql)

        if len(good_plans) == 0:
            if len(plan_with_sqls) > 0:
                good_plans = [plan_with_sqls[0][0]]
                good_plan_sqls = [plan_with_sqls[0][1]]
            else:
                good_plans = [plans[0]]
                good_plan_sqls = ["NO SQL"] 
        
        sample['prompt_planner'] = [prompt_planner] * len(good_plans) 
        sample['planners'] = good_plans
        sample['predict_sqls'] = good_plan_sqls
        # print(re.sub("\s+", " ", good_plan_sqls[0]))
        return sample
    
    def generate_feedbacks(self, sample):
        # key to extend to the same length
        prompt_planner = sample['prompt_planner']
        planners = sample['planners']
        predict_sqls = sample['predict_sqls']

        sample['prompt_planner'] = []
        sample['planners'] = []
        sample['predict_sqls'] = []

        sample['prompt_feedback_select'] = []
        sample['prompt_feedback_condition'] = []
        sample['prompt_feedback_join'] = []
        sample['prompt_feedback_order'] = []
        sample['feedback_selects'] = []
        sample['feedback_conditions'] = []
        sample['feedback_joins'] = []
        sample['feedback_orders'] = []
        sample['pred_results'] = []
        sample['first_try_has_errors'] = []
        
        for prompt_planner, planner, plan_sql in zip(prompt_planner, planners, predict_sqls):
            copy_sample = deepcopy(sample)
            copy_sample['predict_sql'] = plan_sql.replace('\n', ' ')

            # First, get execution_result by executing the SQL query
            execution_result = _execute_sql("./" + sample['db_path'], copy_sample['predict_sql'])

            # Now, call validators in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                # Add tasks conditionally based on the skip flags
                if not args.skip_validator_select:
                    futures.append(executor.submit(self.validator_select.validate, copy_sample, execution_result=execution_result))
                else:
                    futures.append(executor.submit(lambda: ("", "", None)))

                if not args.skip_validator_condition:
                    futures.append(executor.submit(self.validator_condition.validate, copy_sample, execution_result=execution_result))
                else:
                    futures.append(executor.submit(lambda: ("", "", None)))

                if not args.skip_validator_join:
                    futures.append(executor.submit(self.validator_join.validate, copy_sample, execution_result=execution_result))
                else:
                    futures.append(executor.submit(lambda: ("", "", None)))

                if not args.skip_validator_order:
                    futures.append(executor.submit(self.validator_order.validate, copy_sample, execution_result=execution_result))
                else:
                    futures.append(executor.submit(lambda: ("", "", None)))

                # Collect results
                results = [f.result() for f in futures]

            # Unpack the results
            prompt_feedback_select, feedback_selects, _ = results[0]
            prompt_feedback_condition, feedback_conditions, _ = results[1]
            prompt_feedback_join, feedback_joins, _ = results[2]
            prompt_feedback_order, feedback_orders, _ = results[3]


            max_length_feedback = max(len(feedback_selects), len(feedback_conditions), len(feedback_joins), len(feedback_orders))
            # if any feedback is empty, fill with None
            if len(feedback_selects) == 0:
                feedback_selects = [None] * max_length_feedback
            if len(feedback_conditions) == 0:
                feedback_conditions = [None] * max_length_feedback
            if len(feedback_joins) == 0:
                feedback_joins = [None] * max_length_feedback
            if len(feedback_orders) == 0:
                feedback_orders = [None] * max_length_feedback

            # if any feedback has length 1, fill with the first element
            if len(feedback_selects) == 1:
                feedback_selects = feedback_selects * max_length_feedback
            if len(feedback_conditions) == 1:
                feedback_conditions = feedback_conditions * max_length_feedback
            if len(feedback_joins) == 1:
                feedback_joins = feedback_joins * max_length_feedback
            if len(feedback_orders) == 1:
                feedback_orders = feedback_orders * max_length_feedback

            copy_sample['feedback_select'] = feedback_selects
            copy_sample['feedback_condition'] = feedback_conditions
            copy_sample['feedback_join'] = feedback_joins
            copy_sample['feedback_order'] = feedback_orders

            sample['prompt_planner'].extend([prompt_planner] * max_length_feedback)
            sample['planners'].extend([planner] * max_length_feedback)
            sample['predict_sqls'].extend([plan_sql] * len(feedback_selects))

            sample['prompt_feedback_select'].extend([prompt_feedback_select] * len(feedback_selects))
            sample['prompt_feedback_condition'].extend([prompt_feedback_condition] * len(feedback_conditions))
            sample['prompt_feedback_join'].extend([prompt_feedback_join] * len(feedback_joins))
            sample['prompt_feedback_order'].extend([prompt_feedback_order] * len(feedback_orders))
            sample['feedback_selects'].extend(feedback_selects)
            sample['feedback_conditions'].extend(feedback_conditions)
            sample['feedback_joins'].extend(feedback_joins)
            sample['feedback_orders'].extend(feedback_orders)
            sample['pred_results'].extend([_make_str_response(*execution_result)] * max_length_feedback)
            sample['first_try_has_errors'].extend([execution_result[1]] * max_length_feedback)

            assert len(sample['predict_sqls']) == len(sample['feedback_selects'])
            assert len(sample['predict_sqls']) == len(sample['feedback_conditions'])
            assert len(sample['predict_sqls']) == len(sample['feedback_joins'])
            assert len(sample['predict_sqls']) == len(sample['feedback_orders'])
            assert len(sample['predict_sqls']) == len(sample['pred_results'])

        return sample
    
    def generate_fixes(self, sample):

        sample['prompt_fix'] = []
        sample['fixed_sqls'] = []

        temp_prompt_planner = []
        temp_planners = []
        temp_predict_sqls = []

        temp_prompt_selects = []
        temp_prompt_conditions = []
        temp_prompt_joins = []
        temp_prompt_orders = []

        temp_feedback_selects = []
        temp_feedback_conditions = []
        temp_feedback_joins = []
        temp_feedback_orders = []
        temp_pred_results = []
        temp_first_try_has_errors = []

        for i in range(len(sample['predict_sqls'])):
            # process feedback
            select_correct = sample['feedback_selects'][i] is None or 'Conclude: incorrect' not in sample['feedback_selects'][i]
            condition_correct = sample['feedback_conditions'][i] is None or 'Conclude: incorrect' not in sample['feedback_conditions'][i]
            join_correct = sample['feedback_joins'][i] is None or 'Conclude: incorrect' not in sample['feedback_joins'][i]
            order_correct = sample['feedback_orders'][i] is None or 'Conclude: incorrect' not in sample['feedback_orders'][i]
            first_try_has_error = sample['first_try_has_errors'][i]

            if args.skip_validator_select:
                select_correct = True
            if args.skip_validator_join:
                join_correct = True
            if args.skip_validator_condition:
                condition_correct = True
            if args.skip_validator_order:
                order_correct = True

            if first_try_has_error:
                condition_correct = False

            if select_correct and condition_correct and join_correct and order_correct and not first_try_has_error:
                prompt_fixed_sql = None
                fixed_sqls = [None]
            
            else:
                feedback_select = self.validator_select.process_feedback_message_from_completion(sample['prompt_feedback_select'][i], sample['feedback_selects'][i])
                feedback_condition = self.validator_condition.process_feedback_message_from_completion(sample['prompt_feedback_condition'][i], sample['feedback_conditions'][i])
                feedback_join = self.validator_join.process_feedback_message_from_completion(sample['prompt_feedback_join'][i], sample['feedback_joins'][i])
                feedback_order = self.validator_order.process_feedback_message_from_completion(sample['prompt_feedback_order'][i], sample['feedback_orders'][i])

                if select_correct:
                    feedback_select = ""
                if condition_correct:
                    feedback_condition = ""
                if join_correct: 
                    feedback_join = ""
                if order_correct:
                    feedback_order = ""

                copy_sample = deepcopy(sample)
                copy_sample['predict_sql'] = sample['predict_sqls'][i]
                copy_sample['pred_result'] = sample['pred_results'][i]

                prompt_fixed_sql, fixed_sqls = self.fixed_sql_agent.generate(copy_sample, feedback_select, feedback_condition, feedback_join, feedback_order)

                fixed_sqls = [extract_sql_in_code_block(x) for x in fixed_sqls]

                # check executable fixed_sqls
                if args.mode == 'test':
                    filter_fixed_sqls = []
                    for fixed_sql in fixed_sqls:
                        execution_error = check_sql_executability(fixed_sql, sample["db_path"])
                        if execution_error is not None:
                            continue
                        filter_fixed_sqls.append(fixed_sql)
                    if len(filter_fixed_sqls) == 0:
                        fixed_sqls = fixed_sqls[:1]

            sample['prompt_fix'].extend([prompt_fixed_sql] * len(fixed_sqls))
            sample['fixed_sqls'].extend(fixed_sqls)

            temp_prompt_planner.extend([sample['prompt_planner'][i]] * len(fixed_sqls))
            temp_planners.extend([sample['planners'][i]] * len(fixed_sqls))
            temp_predict_sqls.extend([sample['predict_sqls'][i]] * len(fixed_sqls))

            temp_prompt_selects.extend([sample['prompt_feedback_select'][i]] * len(fixed_sqls))
            temp_prompt_conditions.extend([sample['prompt_feedback_condition'][i]] * len(fixed_sqls))
            temp_prompt_joins.extend([sample['prompt_feedback_join'][i]] * len(fixed_sqls))
            temp_prompt_orders.extend([sample['prompt_feedback_order'][i]] * len(fixed_sqls))

            temp_feedback_selects.extend([sample['feedback_selects'][i]] * len(fixed_sqls))
            temp_feedback_conditions.extend([sample['feedback_conditions'][i]] * len(fixed_sqls))
            temp_feedback_joins.extend([sample['feedback_joins'][i]] * len(fixed_sqls))
            temp_feedback_orders.extend([sample['feedback_orders'][i]] * len(fixed_sqls))
            temp_pred_results.extend([sample['pred_results'][i]] * len(fixed_sqls))
            temp_first_try_has_errors.extend([sample['first_try_has_errors'][i]] * len(fixed_sqls))

        sample['prompt_planner'] = temp_prompt_planner
        sample['planners'] = temp_planners
        sample['predict_sqls'] = temp_predict_sqls

        sample['prompt_feedback_select'] = temp_prompt_selects
        sample['prompt_feedback_condition'] = temp_prompt_conditions
        sample['prompt_feedback_join'] = temp_prompt_joins
        sample['prompt_feedback_order'] = temp_prompt_orders

        sample['feedback_selects'] = temp_feedback_selects
        sample['feedback_conditions'] = temp_feedback_conditions
        sample['feedback_joins'] = temp_feedback_joins
        sample['feedback_orders'] = temp_feedback_orders
        sample['pred_results'] = temp_pred_results
        sample['first_try_has_errors'] = temp_first_try_has_errors

        assert len(sample['prompt_planner']) == len(sample['planners'])
        assert len(sample['prompt_planner']) == len(sample['predict_sqls'])
        assert len(sample['prompt_planner']) == len(sample['prompt_feedback_select'])
        assert len(sample['prompt_planner']) == len(sample['prompt_feedback_condition'])
        assert len(sample['prompt_planner']) == len(sample['prompt_feedback_join'])
        assert len(sample['prompt_planner']) == len(sample['prompt_feedback_order'])
        assert len(sample['prompt_planner']) == len(sample['feedback_selects'])
        assert len(sample['prompt_planner']) == len(sample['feedback_conditions'])
        assert len(sample['prompt_planner']) == len(sample['feedback_joins'])
        assert len(sample['prompt_planner']) == len(sample['feedback_orders'])
        assert len(sample['prompt_planner']) == len(sample['pred_results'])
        assert len(sample['prompt_planner']) == len(sample['prompt_fix'])
        assert len(sample['prompt_planner']) == len(sample['fixed_sqls'])
        assert len(sample['prompt_planner']) == len(sample['first_try_has_errors'])

        pair_sqls = [(x, y) for x, y in zip(sample['predict_sqls'], sample['fixed_sqls'])]
        candidate_sqls = [self.fixed_sql_agent.get_final_sql(x, y, sample['db_path']) for x, y in pair_sqls]
        sample['candidate_sqls'] = candidate_sqls

        return sample

    def select_final_sql(self, sample):
        if 'candidate_sqls' not in sample:
            sample['candidate_sqls'] = sample['predict_sqls']

        sample['candidate_pred_results'] = [_execute_sql(sample['db_path'], x)[0] for x in sample['candidate_sqls']]
        sample['final_sql'] = self.selection_agent.get_best_sql(sample, max_candidates=3)
        sample['candidate_pred_results'] = [str(x) for x in sample['candidate_pred_results']]
        return sample


    def generate(self, sample):
        if 'evidence' not in sample:
            sample['evidence'] = ''
        
        if not args.skip_planner:
            sample = self.generate_plans(sample)
        if not args.only_planner:
            if not args.skip_validator:
                sample = self.generate_feedbacks(sample)
            if not args.skip_fix:
                sample = self.generate_fixes(sample)
            if not args.skip_selection:
                sample = self.select_final_sql(sample)
        return sample


def get_answer_planner(messages):
    answers = []
    if args.mode == 'test':
        response = requests.post(f"{args.api_host}/v1/completions",
            json={
                "model": 'planner',
                "prompt": messages[0]['content'],
                "max_tokens": 1024,
                "use_beam_search": False,
                "n": 1,
                "temperature": 0.0,
                "stop": [EOS_TOKEN, '<|end|>', '<|end_header_id|>', '<|end_of_text|>'],
                "seed": args.seed
            }).json()
        answers += [x['text'] for x in response['choices']]

    if args.n_return - 1 > 0:
        response = requests.post(f"{args.api_host}/v1/completions",
            json={
                "model": 'planner',
                "prompt": messages[0]['content'],
                "max_tokens": 1024,
                "use_beam_search": args.use_beam_search,
                "n": args.n_return - 1,
                "temperature": args.temperature,
                "stop": [EOS_TOKEN, '<|end|>', '<|end_header_id|>', '<|end_of_text|>']
            }).json()
        answers += [x['text'] for x in response['choices']]

    # unique answers
    seen = set()
    unique_answers = [x for x in answers if not (x in seen or seen.add(x))]
    return unique_answers

def get_answer_validator(model_name, messages):
    port = int(args.api_host.split(':')[-1])
    api_host = args.api_host.replace(str(port), str(port + 1))
    prompt =  messages[0]['content']
    send_data = {
        "model": 'validator',
        "prompt": prompt,
        "max_tokens": 768,
        "n": args.n_return if args.mode == 'train' else 1,
        "use_beam_search": False,
        "temperature": args.temperature if args.mode == 'train' else 0.0,
        "stop": [EOS_TOKEN, '<|end|>', '<|end_header_id|>'],
        "seed": args.seed
    }
    if args.use_beam_search_validator:
        send_data['use_beam_search'] = True
        send_data['n'] = args.n_return

    response = requests.post(f"{api_host}/v1/completions",
        json=send_data).json()
    answers = []
    for x in response['choices']:
        answers.append(x['text'])

    if args.mode == 'test':
        answers = answers[:1]
    return answers


def get_answer_fixed(messages):
    port = int(args.api_host.split(':')[-1])
    api_host = args.api_host.replace(str(port), str(port + 2))
    response = requests.post(f"{api_host}/v1/completions",
        json={
            "model": 'fixed',
            "prompt": messages[0]['content'],
            "max_tokens": 256,
            "use_beam_search": args.use_beam_search,
            "n": args.n_return if args.mode == 'train' else 1,
            "temperature": args.temperature if args.mode == 'train' else 0.0,
            "stop": [EOS_TOKEN, '<|end|>', '<|end_header_id|>'],
            "seed": args.seed
            }).json()
    answers = [x['text'] for x in response['choices']]
    seen = set()
    unique_answers = [x for x in answers if not (x in seen or seen.add(x))]
    return unique_answers

def get_answer_selection(messages):
    port = int(args.api_host.split(':')[-1])
    api_host = args.api_host.replace(str(port), str(port + 3))
    response = requests.post(f"{api_host}/v1/completions",
        json={
            "model": 'selection',
            "prompt": messages[0]['content'],
            "max_tokens": 8,
            "use_beam_search": False,
            "n": 1,
            "temperature": 0.0,
            "stop": [ '<|eot_id|>', '<|end|>', '<|end_header_id|>', '<|end_of_text|>', '<｜end▁of▁sentence｜>']
        }).json()
    answers = [x['text'] for x in response['choices']]
    return answers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../data/sft_bird_with_evidence_train_text2sql.json')
    parser.add_argument('--output_file', type=str, default='../data/planner/planner_select_bird_with_evidence_train.jsonl')
    parser.add_argument('--model-name', type=str, default='phi', choices=['phi', 'llama', 'codes', 'qwen', 'nl2sql'])
    parser.add_argument('--use_beam_search', action='store_true')
    parser.add_argument('--n_return', type=int, default=1, help="Number of responses to return for each agent. While the number of agents is 3, the total number of responses will be n_return ** 3")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--api_host', default='http://localhost:8001', type=str)
    parser.add_argument('--mode', default='test', choices=['test', 'train'])
    parser.add_argument('--only_planner', action='store_true')
    parser.add_argument('--skip_planner', action='store_true')
    parser.add_argument('--skip_validator', action='store_true')
    parser.add_argument('--skip_fix', action='store_true')
    parser.add_argument('--skip_validator_select', action='store_true')
    parser.add_argument('--skip_validator_condition', action='store_true')
    parser.add_argument('--skip_validator_join', action='store_true')
    parser.add_argument('--skip_validator_order', action='store_true', default=True)
    parser.add_argument('--skip_selection', action='store_true')
    parser.add_argument('--n_processes', default=64, type=int)
    parser.add_argument('--use_beam_search_validator', action='store_true')
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    return args

import os
import sys
import json
import traceback
from multiprocessing import Pool, Manager
from tqdm import tqdm

def init_worker():
    global mas
    mas = MultiAgentSystem(None)

def process_sample(args):
    sample, output_file_path = args
    try:
        # sample['schema_sequence'] = sample['schema_sequence'].replace('; values:', '; example values:')
        sample = mas.generate(sample)
        # Write to file directly with synchronization
        with lock:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                f.flush()
    except Exception as e:
        # Re-raise the exception to be caught in the main process
        traceback.print_exc()
        raise e

def update_data_with_old_output(args, data):
    if os.path.exists(args.output_file):
        old_output = {}
        # Load the old output file and store it in a dictionary for quick lookups
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    key = f"{sample['source']} {sample['db_id']} {sample['question']}"
                    old_output[key] = sample
                except Exception as err:
                    print(err)

        # Replace data entries with corresponding entries from old_output
        for i, sample in enumerate(data):
            key = f"{sample['source']} {sample['db_id']} {sample['question']}"
            if key in old_output:
                data[i] = old_output[key]

        # Rewrite old_output to output_file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for sample in old_output.values():
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        old_output = {}

    # unique_data by keys
    unique_data = {}
    for i, sample in enumerate(data):
        key = f"{sample['source']} {sample['db_id']} {sample['question']}"
        unique_data[key] = sample
    print("unique_data", len(unique_data))

    # Remove already processed entries from data
    data = [sample for sample in data if f"{sample['source']} {sample['db_id']} {sample['question']}" not in old_output]

    return data

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    if args.model_name == 'phi':
        EOS_TOKEN = '<|end|>'
        ASSISTANT_TOKEN = '<|assistant|>'
        USER_TOKEN = '<|user|>'
    elif 'llama' in args.model_name:
        EOS_TOKEN = '<|eot_id|>'
        ASSISTANT_TOKEN = '<|start_header_id|>assistant<|end_header_id|>'
        USER_TOKEN = '<|start_header_id|>user<|end_header_id|>'
    elif 'codes' in args.model_name:
        EOS_TOKEN = '<|eot_id|>'
        ASSISTANT_TOKEN = '<|assistant|>'
        USER_TOKEN = '<|user|>'
    elif args.model_name == 'qwen':
        EOS_TOKEN = '<|im_end|>'
        ASSISTANT_TOKEN = '<|im_start|>assistant'
        USER_TOKEN = '<|im_start|>user'
    elif args.model_name == 'nl2sql':
        EOS_TOKEN = '<|eot_id|>'
        ASSISTANT_TOKEN = '<|start_header_id|>assistant<|end_header_id|>'
        USER_TOKEN = '<|start_header_id|>user<|end_header_id|>'
    else:
        raise Exception('Invalid model name')
    
    data = json.load(open(args.input_file, 'r', encoding='utf-8'))
    print(len(data))
    data = update_data_with_old_output(args, data)

    # Make directories if they do not exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    manager = Manager()
    lock = manager.Lock()
    args_list = [(sample, args.output_file) for sample in data]

    if len(args_list) == 0:
        import sys; sys.exit()

    with Pool(processes=args.n_processes, initializer=init_worker) as pool:
        results = []
        for params in args_list:
            res = pool.apply_async(process_sample, args=(params,))
            results.append(res)

        # Use tqdm to display progress
        for res in tqdm(results):
            try:
                res.get()
            except Exception as e:
                # Print the traceback of the exception
                print("An error occurred:", file=sys.stderr)
                traceback.print_exc()
                pool.terminate()
                pool.join()
                sys.exit(1)
