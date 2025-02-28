import json
from tqdm import tqdm
from validator_data.validator import ValidatorSelect, ValidatorJOIN, ValidatorOrder, _execute_sql, _make_str_response, ValidatorCondition, get_answer_openai
from data_processing.planner import is_execution_correct
import re
import os
from datasets import Dataset, DatasetDict
from multiprocessing import Pool
import numpy as np
from data_processing.utils import norm_sql_query

def extract_sql_in_code_block(pred_sql_text):
    """
    Extracts the SQL query from a text block that contains code block marked by triple backticks (```sql ... ```).
    
    Args:
        pred_sql_text (str): The input text that may contain a SQL code block.
    
    Returns:
        str: The extracted SQL query or an empty string if no SQL code block is found.
    """
    # Use regex to search for the SQL code block enclosed in triple backticks
    sql_block_match = re.search(r"```(.+?)```", pred_sql_text, re.DOTALL)

    if sql_block_match:
        # Extract the SQL query from the matched block
        sql_query = sql_block_match.group(1).strip()
        if sql_query.startswith("sql"):
            sql_query = sql_query.replace("sql", "")
        # print('extract: ', sql_query)
        return sql_query
    else:
        return pred_sql_text


def get_final_predict_sql(sample):
    if 'fixed_sqls' in sample:
        predict_sqls = [x for x in sample['fixed_sqls']]
        predict_sqls = [x for x in predict_sqls if x is not None]
        predict_sqls = [extract_sql_in_code_block(x) for x in predict_sqls]

        for i in range(len(predict_sqls)):
            if predict_sqls[i] is None:
                predict_sqls[i] = sample['predict_sqls'][i]
    else:
        predict_sqls = sample['predict_sqls']

    return predict_sqls

def get_predict_sql_from_planner(plan):
    pred_sql_match = re.search(r"(?<=Final SQL query:).*?```(.*?)```", plan, re.DOTALL)
    if pred_sql_match is None:
        pred_sql = " "
    else:
        # Extract the SQL query from within the backticks
        pred_sql = pred_sql_match.group(1).strip()
    return pred_sql

def get_positive_samples_and_negative_samples(agent_data):
    """
    1. group the agent_data by the prompt
    2. for each prompt, if there are multiple completions, then the positive sample is the completion that is correct
    3. the negative samples are the completions that are incorrect
    """
    prompt_to_completions = {}
    for data in agent_data:
        prompt = data['prompt']
        completion = data['completion']
        reward = data['reward']
        db_path = data.get('db_path', None)

        if prompt not in prompt_to_completions:
            prompt_to_completions[prompt] = []

        # completion = '\n' + completion.strip()
        prompt_to_completions[prompt].append((completion, reward, db_path))

    dpo_data = [] # each element is a dictionary with keys: prompt, chosen, rejected
    for prompt, completions in prompt_to_completions.items():
        dpo_sample = {
            'prompt': prompt,
            'chosen': [],
            'rejected': [],
            'db_path': completions[0][2] if len(completions) > 0 else None
        }
        
        for completion, reward, _ in completions:
            if reward == 1:
  #              completion = '\n' + completion.strip()
                dpo_sample['chosen'].append(completion)
            else:
 #               completion = '\n' + completion.strip()
                if len(completion.split()) < 10: continue
                dpo_sample['rejected'].append(completion)

        chosen = dpo_sample['chosen']
        rejected = dpo_sample['rejected']
        dpo_sample['rejected'] = [x for x in dpo_sample['rejected'] if x not in chosen]
        dpo_sample['chosen'] = [x for x in dpo_sample['chosen'] if x not in rejected]
        
      #  if len(dpo_sample['rejected']) == 0:
      #      dpo_sample['rejected'].append("")

        if len(dpo_sample['chosen']) > 0 and len(dpo_sample['rejected']) > 0:
            dpo_data.append(dpo_sample)
    return dpo_data

def build_dpo_ranking_data(agent_data):
    prompt_to_completions = {}
    for data in agent_data:
        prompt = data['prompt']
        completion = data['completion']
        reward = data['reward']
        db_path = data.get('db_path', None)

        if prompt not in prompt_to_completions:
            prompt_to_completions[prompt] = []

        completion = '\n' + completion.strip()
        prompt_to_completions[prompt].append({
            'completion': completion,
            'reward': reward,
            'db_path': db_path,
            'question': data['question'],
            'db_id': data['db_id'],
            'sql': data['sql']
        })

    dpo_data = [] # each element is a dictionary with keys: prompt, chosen, rejected
    for prompt, completions in prompt_to_completions.items():

        completions = sorted(completions, key=lambda x: x['reward'], reverse=True)
        reward_1s = [x for x in completions if x['reward'] == 1]
        reward_0s = [x for x in completions if x['reward'] == 0]
        reward_ns = [x for x in completions if x['reward'] == -1]

        chosens = reward_1s
        rejecteds = reward_ns + reward_0s

        # if len(chosens) == 0:
        #     chosens.append({
        #         "completion": completions[0]['sql'],
        #         "reward": 1.0
        #     })

        np.random.shuffle(chosens)
        # np.random.shuffle(reward_0s)
        # np.random.shuffle(reward_ns)
        # np.random.shuffle(rejecteds)

        # Ensure we have at least one rejected entry to form pairs
        n_select = args.n_select_chosens
        if len(rejecteds) < n_select:
            rejecteds += [{
                'completion': "",
                'reward': -2
            }] * (n_select - len(rejecteds))

        assert len(rejecteds) >= n_select

        # Form pairs by combining `chosens` and `rejecteds`
        for chosen, rejected in zip(chosens[:n_select], rejecteds[:n_select]):
            dpo_sample = {
                'prompt': prompt,
                'chosen': chosen['completion'],
                'rejected': rejected['completion'],
                'question': completions[0]['question'],
                'db_id': completions[0]['db_id']
            }
            dpo_data.append(dpo_sample)
        # for chosen in chosens:
        #     for rejected in rejecteds:
        #         dpo_sample = {
        #             'prompt': prompt,
        #             'chosen': chosen['completion'],
        #             'rejected': rejected['completion'],
        #             'question': completions[0]['question'],
        #             'db_id': completions[0]['db_id']
        #         }
        #         dpo_data.append(dpo_sample)

    return dpo_data


def get_fix_sql_from_openai(prompt_fix):
    prompt_fix = prompt_fix.replace("<|start_header_id|>user<|end_header_id|>", "").replace("<|start_header_id|>assistant<|end_header_id|>", "").replace("<|eot_id|>", "").strip()

    prompt_fix = """You are SQL Tutor that fixes the student query. Given a database schema, a question, and SQL query generated by student, its response in database and the feedback on the correctness of the query. Based on the Feedback, generate a fixed sql that correctly aligns with the intent of question.\nGenerate SQL query directly without explanation.\n""" + prompt_fix

    messages = [
        {
            'role': 'user',
            'content': prompt_fix
        }
    ]

    answer = get_answer_openai(CLIENT, messages)[0]
    return answer

def process_prompt_fix(prompt_fix):
    if prompt_fix is None:
        return None
    answer_openai = get_fix_sql_from_openai(prompt_fix)
    gpt_fixed_sql = extract_sql_in_code_block(answer_openai)
    return gpt_fixed_sql

from concurrent.futures import ThreadPoolExecutor

# Define a helper function for executing SQL with a given path
def execute_sql_with_path(args):
    db_path, sql = args
    if sql is None:
        return None, None
    return _execute_sql("./" + db_path, sql)

def is_the_same_sql(sql1, sql2, db_path):
    # norm \s+ to " " and strip
    # sql1 = re.sub(r'\s+', ' ', sql1).strip().lower()
    # sql2 = re.sub(r'\s+', ' ', sql2).strip().lower()
    # # remove ``
    # sql1 = sql1.replace('`', '')
    # sql2 = sql2.replace('`', '')

    # execute and check if is_execution_correct
    sql1 = _execute_sql("./" + db_path, sql1)[0]
    sql2 = _execute_sql("./" + db_path, sql2)[0]
    return is_execution_correct(sql1, sql2)

def is_valid_feedback(feedback):
    if 'Conclude: correct' in feedback or 'Conclude: incorrect' in feedback:
        return True
    return False

import requests
def get_answer_fixed(prompt):
    response = requests.post(f"http://localhost:8005/v1/completions",
        json={
            "model": 'fixed',
            "prompt": prompt,
            "max_tokens": 200,
            "use_beam_search": False,
            "n": 1,
            "temperature": 0.0,
            "stop": [EOS_TOKEN, '<|end|>', '<|end_header_id|>']
            }).json()
    answers = [x['text'] for x in response['choices']]
    return answers


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


def is_better_than_previous_response(sample, true_execution, feedback_type='feedback_selects'):
    # return True
    # check if modified feedback is better than previous feedback
    if "Conclude: correct" not in sample[f'modified_{feedback_type}'][0] and "Conclude: incorrect" not in sample[f'modified_{feedback_type}'][0]:
        return False

    if f'modified_{feedback_type}' not in sample:
        return False
    
    modified_feedback = sample[f'modified_{feedback_type}'][0]
    previous_feedback = sample[feedback_type][0]

    if previous_feedback is None:
        return False

    modified_feedback_conclude_correct = "Conclude: correct" in modified_feedback
    previous_feedback_conclude_correct = "Conclude: correct" in previous_feedback
    # if different conclusion then return False
    if modified_feedback_conclude_correct != previous_feedback_conclude_correct:
        return False

    feedback_select = sample['feedback_selects'][0]
    feedback_condition = sample['feedback_conditions'][0]
    feedback_join = sample['feedback_joins'][0]
    # feedback_order = sample['feedback_orders'][0]

    if feedback_type == 'feedback_selects':
        feedback_select = modified_feedback
    elif feedback_type == 'feedback_conditions':
        feedback_condition = modified_feedback
    elif feedback_type == 'feedback_joins':
        feedback_join = modified_feedback
    # elif feedback_type == 'feedback_orders':
    #     feedback_order = modified_feedback

    select_correct = feedback_select is None or 'Conclude: correct' in feedback_select
    condition_correct = feedback_condition is None or 'Conclude: correct' in feedback_condition
    join_correct = feedback_join is None or 'Conclude: correct' in feedback_join
    # order_correct = feedback_order is None or 'Conclude: correct' in feedback_order

    if select_correct:
        feedback_select = ""
    if condition_correct:
        feedback_condition = ""
    if join_correct: 
        feedback_join = ""
    # if order_correct:
    #     feedback_order = ""
    
    new_prompt_fix = PROPMT_FIX.format(
        schema=sample['schema_sequence'],
        question=sample['question'],
        evidence=sample['evidence'],
        sql_query=sample['predict_sqls'][0],
        execution_response=sample['pred_result'][0],
        feedback_select=feedback_select,
        feedback_condition=feedback_condition,
        feedback_join=feedback_join,
        # feedback_order=feedback_order
    )
    fixed_sql = get_answer_fixed(new_prompt_fix)[0]
    print(fixed_sql)

    new_execution = _execute_sql("./" + sample['db_path'], fixed_sql)

    if is_execution_correct(true_execution[0], new_execution[0]):
        return True
    return False

def is_bad_feedback(feedback):
    # if feedback contains many conclution, then return 0
    if feedback is None or len(feedback.split("Conclude:")) > 2:
        return True
    return False

def concat_db_response_to_completion(completion, execution_response):
    has_error = execution_response[1]
    if has_error:
        completion = completion + "\nResponse: " + execution_response[0]
    else:
        completion = completion + "\nResponse: No error"
    return completion

def process_sample(sample):
    true_execution_result = _execute_sql("./" + sample['db_path'], sample['sql'])

    # Get the predicted SQL queries
    predict_sqls = get_final_predict_sql(sample)

    #
    fixed_sqls = [x for x in sample.get('fixed_sqls', [])]
    fixed_sqls = [x for x in fixed_sqls if x is not None]
    fixed_sqls = [extract_sql_in_code_block(x) for x in fixed_sqls]
    planner_sqls = sample['predict_sqls']

    with ThreadPoolExecutor(max_workers=8) as executor:
        planner_sql_execution_results = list(executor.map(execute_sql_with_path, [(sample['db_path'], sql) for sql in planner_sqls]))   

    is_planner_sql_execution_corrects = [
        is_execution_correct(true_execution_result[0], execution_result[0])
        for execution_result in planner_sql_execution_results
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        fixed_sql_execution_results = list(executor.map(execute_sql_with_path, [(sample['db_path'], sql) for sql in fixed_sqls]))
    is_fixed_sql_execution_corrects = [
        is_execution_correct(true_execution_result[0], execution_result[0])
        for execution_result in fixed_sql_execution_results
    ]

    # # Prepare arguments for multi-threading
    # sql_execution_args = [(sample['db_path'], sql) for sql in predict_sqls]

    # # Use multi-threading to execute predicted SQLs
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     execution_results = list(executor.map(execute_sql_with_path, sql_execution_args))
    
    # is_execution_corrects = [
    #     is_execution_correct(true_execution_result[0], execution_result[0]) 
    #     for execution_result in execution_results
    # ]

    # Initialize data lists
    planner_data = [] # prompt, completion, reward (0 or 1)
    validator_select_data = [] # prompt, completion, reward (0 or 1)
    validator_condition_data = [] # prompt, completion, reward (0 or 1)
    validator_join_data = [] # prompt, completion, reward (0 or 1)
    validator_order_data = [] # prompt, completion, reward (0 or 1)
    fixed_sql_data = [] # prompt, completion, reward (0 or 1)

    # Process GPT planner and check correctness
    if gpt_question2planner is not None:
        if sample['question'] in gpt_question2planner:
            gpt_plan = gpt_question2planner[sample['question']]
            gpt_predict_sql = get_predict_sql_from_planner(gpt_plan)
            gpt_execution_result = _execute_sql("./" + sample['db_path'], gpt_predict_sql)

            completion = gpt_question2planner[sample['question']]
            #completion = concat_db_response_to_completion(completion, gpt_execution_result)

            if is_execution_correct(true_execution_result[0], gpt_execution_result[0]):
                planner_data.append({
                    'prompt': sample['prompt_planner'][0],
                    'completion': completion,
                    'reward': 1,
                    'db_path': sample['db_path'],
                    'db_id': sample['db_id'],
                    'question': sample['question'],
                     'sql': norm_sql_query(sample['sql'], sample['schema'])
                })
            # true_response, true_sql_has_error = true_execution_result
            # pred_response, pred_sql_has_error = gpt_execution_result
            # if (not true_sql_has_error) and pred_sql_has_error:
            #     planner_data.append({
            #         'question': sample['question'],
            #         'db_id': sample['db_id'],
            #         'prompt': sample['prompt_planner'][0],
            #         'completion': completion,
            #         'reward': -1,
            #         'db_path': sample['db_path'],
            #         'sql': norm_sql_query(sample['sql'], sample['schema'])
            #     })


    # Process fixed SQLs if present
    if 'prompt_fix' in sample:
        prompt_fixs = sample['prompt_fix'][:1]

        if args.enable_advanced_fix_agent:
            gpt_fixed_sqls = [process_prompt_fix(prompt_fix) for prompt_fix in prompt_fixs]

            # prompt2sql = open('prompt2sql.json').read()
            # prompt2sql = json.loads(prompt2sql)
            # gpt_fixed_sqls = [prompt2sql[prompt_fix] if prompt_fix in prompt2sql else None for prompt_fix in prompt_fixs]
            
            # Use multi-threading for fixed SQL execution
            fixed_sql_args = [(sample['db_path'], sql) for sql in gpt_fixed_sqls]
            with ThreadPoolExecutor(max_workers=32) as executor:
                gpt_execution_results = list(executor.map(execute_sql_with_path, fixed_sql_args))
            
            for prompt_fix, gpt_fixed_sql, exec_result in zip(prompt_fixs, gpt_fixed_sqls, gpt_execution_results):
                if gpt_fixed_sql is not None:
                    fixed_sql_data.append({
                        'prompt': prompt_fix,
                        'completion': gpt_fixed_sql,
                        'reward': int(is_execution_correct(true_execution_result[0], exec_result[0]))
                    })

    # norm prompt_feedback and feedbacks
    # if 'is_correct' in sample and ((not sample['is_correct']) and sample['select_correct'] and sample['condition_correct'] and sample['join_correct'] and sample['order_correct']):
    if True:
        if "prompt_feedback_select" in sample:
            for i in range(len(sample['prompt_feedback_select'])):
                if sample['prompt_feedback_select'][i] is None:
                    continue

                feedback_select = sample['feedback_selects'][i]
                prompt_completion = sample['prompt_feedback_select'][i] + feedback_select

                prompt_before_feedback_token = prompt_completion.split("SELECT.")[0].strip()  + "\n" + "SELECT."
                feedback_select = prompt_completion.split("SELECT.")[1]

                sample['prompt_feedback_select'][i] = prompt_before_feedback_token
                sample['feedback_selects'][i] = feedback_select
        
        if "prompt_feedback_condition" in sample:
            for i in range(len(sample['prompt_feedback_condition'])):
                if sample['prompt_feedback_condition'][i] is None:
                    continue

                feedback_condition = sample['feedback_conditions'][i]
                prompt_before_feedback_token = sample['prompt_feedback_condition'][i].split("CONDITION.")[0].strip() + "\n" + "CONDITION."
                feedback_condition = sample['prompt_feedback_condition'][i].split("CONDITION.")[1].strip() + "\n" + feedback_condition.strip()

                sample['prompt_feedback_condition'][i] = prompt_before_feedback_token
                sample['feedback_conditions'][i] = feedback_condition
        
        if "prompt_feedback_join" in sample:
            for i in range(len(sample['prompt_feedback_join'])):
                if sample['prompt_feedback_join'][i] is None:
                    continue

                feedback_join = sample['feedback_joins'][i]
                prompt_before_feedback_token = sample['prompt_feedback_join'][i].split("JOIN.")[0].strip() + "\n" + "JOIN."
                feedback_join = sample['prompt_feedback_join'][i].split("JOIN.")[1] + feedback_join.strip()

                sample['prompt_feedback_join'][i] = prompt_before_feedback_token
                sample['feedback_joins'][i] = feedback_join

        # if "prompt_feedback_order" in sample:
        #     for i in range(len(sample['prompt_feedback_order'])):
        #         if sample['prompt_feedback_order'][i] is None:
        #             continue
        
        #         feedback_order = sample['feedback_orders'][i]
        #         prompt_before_feedback_token = sample['prompt_feedback_order'][i].split("ORDER BY.")[0].strip() + "\n" + "ORDER BY."
        #         feedback_order = sample['prompt_feedback_order'][i].split("ORDER BY.")[1]+ feedback_order.strip()
                
        #         sample['prompt_feedback_order'][i] = prompt_before_feedback_token
        #         sample['feedback_orders'][i] = feedback_order


        if "modified_feedback_selects" in sample:
            if is_valid_feedback(sample['modified_feedback_selects'][0]):
                if is_better_than_previous_response(sample, true_execution_result, feedback_type='feedback_selects') and sample['prompt_feedback_select'][0] is not None:
                    validator_select_data.append({
                        'prompt': sample['prompt_feedback_select'][0],
                        'completion': sample['modified_feedback_selects'][0],
                        'reward': 1
                    })
                    validator_select_data.append({
                        'prompt': sample['prompt_feedback_select'][0],
                        'completion': sample['feedback_selects'][0],
                        'reward': 0
                    })
        if "modified_feedback_conditions" in sample:
            if is_valid_feedback(sample['modified_feedback_conditions'][0]):
                if is_better_than_previous_response(sample, true_execution_result, feedback_type='feedback_conditions') and sample['prompt_feedback_condition'][0] is not None:
                    validator_condition_data.append({
                        'prompt': sample['prompt_feedback_condition'][0],
                        'completion': sample['modified_feedback_conditions'][0],
                        'reward': 1
                    })
                    validator_condition_data.append({
                        'prompt': sample['prompt_feedback_condition'][0],
                        'completion': sample['feedback_conditions'][0],
                        'reward': 0
                    })     
        if "modified_feedback_joins" in sample:
            if is_valid_feedback(sample['modified_feedback_joins'][0]):
                if is_better_than_previous_response(sample, true_execution_result, feedback_type='feedback_joins') and sample['prompt_feedback_join'][0] is not None:
                    validator_join_data.append({
                        'prompt': sample['prompt_feedback_join'][0],
                        'completion': sample['modified_feedback_joins'][0],
                        'reward': 1
                    })
                    validator_join_data.append({
                        'prompt': sample['prompt_feedback_join'][0],
                        'completion': sample['feedback_joins'][0],
                        'reward': 0
                    })
        # if "modified_feedback_orders" in sample:
        #     if is_valid_feedback(sample['modified_feedback_orders'][0]):
        #         if is_better_than_previous_response(sample, true_execution_result, feedback_type='feedback_orders') and sample['prompt_feedback_order'][0] is not None:
        #             validator_order_data.append({
        #                 'prompt': sample['prompt_feedback_order'][0],
        #                 'completion': sample['modified_feedback_orders'][0],
        #                 'reward': 1
        #             })
        #             validator_order_data.append({
        #                 'prompt': sample['prompt_feedback_order'][0],
        #                 'completion': sample['feedback_orders'][0],
        #                 'reward': 0
        #             })
            
        # print(len(validator_condition_data))

    for i, (plan, is_correct, execution_result) in enumerate(zip(sample['planners'], is_planner_sql_execution_corrects, planner_sql_execution_results)):
        #completion = concat_db_response_to_completion(plan, execution_result)
        # planner_data.append({
        #     'prompt': sample['prompt_planner'][i],
        #     'completion': plan,
        #   #  'reward': int(is_correct),
        #     'reward': 1 if (not execution_result[1]) else 0,
        #     'db_path': sample['db_path']
        # })

        pred_response, pred_sql_has_error = execution_result
        if pred_sql_has_error:
            planner_data.append({
                'question': sample['question'],
                'db_id': sample['db_id'],
                'prompt': sample['prompt_planner'][i],
                'completion': plan,
                'reward': -1,
                'db_path': sample['db_path'],
                'sql': norm_sql_query(sample['sql'], sample['schema'])
            })
        else:
            planner_data.append({
                'question': sample['question'],
                'db_id': sample['db_id'],
                'prompt': sample['prompt_planner'][i],
                'completion': plan,
                'reward': int(is_correct),
                'db_path': sample['db_path'],
                'sql': norm_sql_query(sample['sql'], sample['schema'])
            }) 

        if 'feedback_selects' not in sample:
            continue
        # add validator data 
        select_correct = sample['feedback_selects'][i] is None or 'Conclude: correct' in sample['feedback_selects'][i]
        condition_correct = sample['feedback_conditions'][i] is None or 'Conclude: correct' in sample['feedback_conditions'][i]
        join_correct = sample['feedback_joins'][i] is None or 'Conclude: correct' in sample['feedback_joins'][i]
        # order_correct = sample['feedback_orders'][i] is None or 'Conclude: correct' in sample['feedback_orders'][i]

        if is_correct:
            # if not correct then cannot evaluate the correctness of the feedback since each feedback comments only 1 part of the query
            if sample['feedback_selects'][i] not in [x['completion'] for x in validator_select_data]:
                validator_select_data.append({
                    'prompt': sample['prompt_feedback_select'][i],
                    'completion': sample['feedback_selects'][i],
                    'reward': 1.0 if not is_bad_feedback(sample['feedback_selects'][i]) and  select_correct else 0.0
                })
            if sample['feedback_conditions'][i] not in [x['completion'] for x in validator_condition_data]:
                validator_condition_data.append({
                    'prompt': sample['prompt_feedback_condition'][i],
                    'completion': sample['feedback_conditions'][i],
                    'reward': 1.0 if not is_bad_feedback(sample['feedback_conditions'][i]) and condition_correct else 0.0
                })
            if sample['feedback_joins'][i] not in [x['completion'] for x in validator_join_data]:
                validator_join_data.append({
                    'prompt': sample['prompt_feedback_join'][i],
                    'completion': sample['feedback_joins'][i],
                    'reward': 1.0 if not is_bad_feedback(sample['feedback_joins'][i]) and join_correct else 0.0
                })
            # if sample['feedback_orders'][i] not in [x['completion'] for x in validator_order_data]:
            #     validator_order_data.append({
            #         'prompt': sample['prompt_feedback_order'][i],
            #         'completion': sample['feedback_orders'][i],
            #         'reward': 1.0 if not is_bad_feedback(sample['feedback_orders'][i]) and order_correct else 0.0
            #     })

    # Append data for each predicted SQL
    for i, (fixed_sql, is_correct, planner_sql, is_planner_correct) in enumerate(zip(fixed_sqls, is_fixed_sql_execution_corrects, planner_sqls, is_planner_sql_execution_corrects)):
        if fixed_sql is not None:
            fixed_sql_data.append({
                'prompt': sample['prompt_fix'][i],
                'completion': sample['fixed_sqls'][i],
                'reward': 0 if is_the_same_sql(fixed_sql, planner_sql, sample['db_path']) else int(is_correct)
            })
        
        if not is_planner_correct:
            if sample['feedback_selects'][i] not in [x['completion'] for x in validator_select_data]:
                validator_select_data.append({
                    'prompt': sample['prompt_feedback_select'][i],
                    'completion': sample['feedback_selects'][i],
                    'reward': 1.0 if not is_bad_feedback(sample['feedback_selects'][i]) and is_correct else 0.0
                })
            if sample['feedback_conditions'][i] not in [x['completion'] for x in validator_condition_data]:
                validator_condition_data.append({
                    'prompt': sample['prompt_feedback_condition'][i],
                    'completion': sample['feedback_conditions'][i],
                    'reward': 1.0 if not is_bad_feedback(sample['feedback_conditions'][i]) and is_correct else 0.0
                })
            if sample['feedback_joins'][i] not in [x['completion'] for x in validator_join_data]:
                validator_join_data.append({
                    'prompt': sample['prompt_feedback_join'][i],
                    'completion': sample['feedback_joins'][i],
                    'reward': 1.0 if not is_bad_feedback(sample['feedback_joins'][i]) and is_correct else 0.0
                })
            # if sample['feedback_orders'][i] not in [x['completion'] for x in validator_order_data]:
            #     validator_order_data.append({
            #         'prompt': sample['prompt_feedback_order'][i],
            #         'completion': sample['feedback_orders'][i],
            #         'reward': 1.0 if not is_bad_feedback(sample['feedback_orders'][i]) and is_correct else 0.0
            #     })

    # Filter out samples with None values for prompt or completion
    planner_data = [x for x in planner_data if x['prompt'] is not None and x['completion'] is not None]
    validator_select_data = [x for x in validator_select_data if x['prompt'] is not None and x['completion'] is not None]
    validator_condition_data = [x for x in validator_condition_data if x['prompt'] is not None and x['completion'] is not None]
    validator_join_data = [x for x in validator_join_data if x['prompt'] is not None and x['completion'] is not None]
    validator_order_data = [x for x in validator_order_data if x['prompt'] is not None and x['completion'] is not None]
    fixed_sql_data = [x for x in fixed_sql_data if x['prompt'] is not None and x['completion'] is not None]

    # Get positive and negative samples for DPO
    planner_dpo_data = build_dpo_ranking_data(planner_data)
    validator_select_dpo_data = get_positive_samples_and_negative_samples(validator_select_data)
    validator_condition_dpo_data = get_positive_samples_and_negative_samples(validator_condition_data)
    validator_join_dpo_data = get_positive_samples_and_negative_samples(validator_join_data)
    validator_order_dpo_data = get_positive_samples_and_negative_samples(validator_order_data)
    fixed_sql_dpo_data = get_positive_samples_and_negative_samples(fixed_sql_data)

    return {
        'planner': planner_dpo_data,
        'validator_select': validator_select_dpo_data,
        'validator_condition': validator_condition_dpo_data,
        'validator_join': validator_join_dpo_data,
        'validator_order': validator_order_dpo_data,
        'fixed_sql': fixed_sql_dpo_data
    }


def make_hf_dataset(input_file):
    """
    Make a Hugging Face dataset from the input file (dpo-*.jsonl) file.
    """
    samples = []
    added_samples = set()
    with open(input_file) as fp:
        for line in fp:
            line_sample = json.loads(line)
            for sample in line_sample:
                prompt = sample['prompt']
                sample['chosen'] = list(set(sample['chosen']))
                sample['rejected'] = list(set(sample['rejected']))
                min_length = min(len(sample['chosen']), len(sample['rejected']))
                sample['chosen'] = sample['chosen'][:min_length]
                sample['rejected'] = sample['rejected'][:min_length]
                assert type(sample['chosen']) == list, 'error'
                assert type(sample['rejected']) == list, 'error'
                for chosen in sample['chosen']:
                    #chosen = chosen# + EOS_TOKEN
                    for rejected in sample['rejected']:
                       # rejected = rejected# + EOS_TOKEN
                        added_data = {
                            'prompt': prompt,
                            'chosen': chosen,
                            'rejected': rejected
                        }
                        key = f"{prompt} {chosen} {rejected}".strip()
                        if key not in added_samples:
                            samples.append(added_data)
                            added_samples.add(key)

    return DatasetDict({
        'train_dpo': Dataset.from_list(samples),
        'test_dpo': Dataset.from_list(samples[:100])
    })
     
if __name__ == '__main__':
    import os
    import json
    from tqdm import tqdm
    from openai import OpenAI
    from dotenv import load_dotenv
    import argparse
    from multiprocessing import Pool
    import traceback

    # Load environment variables
    load_dotenv()
    CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process files for LLM alignment.")
    parser.add_argument("--input_file", default="data/llm_alignment/llama-3-end2end-spider_train-p5.jsonl", help="Path to the input JSONL file")
    parser.add_argument("--gpt_planner_file", default=None, help="Enable Advanced Planner Agent, this point to a saved data path to the GPT planner file")
    parser.add_argument("--output_planner_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_planner.jsonl", help="Path for the planner output JSONL file")
    parser.add_argument("--output_validator_select_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_validator_select.jsonl", help="Path for the validator select output JSONL file")
    parser.add_argument("--output_validator_condition_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_validator_condition.jsonl", help="Path for the validator condition output JSONL file")
    parser.add_argument("--output_validator_join_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_validator_join.jsonl", help="Path for the validator join output JSONL file")
    parser.add_argument("--output_validator_order_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_validator_order.jsonl", help="Path for the validator order output JSONL file")
    parser.add_argument("--output_fixed_sql_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_fixed_sql.jsonl", help="Path for the fixed SQL output JSONL file")

    parser.add_argument("--enable_advanced_fix_agent", action="store_true", help="Enable the GPT fix agent")
    parser.add_argument("--n_select_chosens", default=2, type=int)
    args = parser.parse_args()

    # Ensure output directories exist
    for path in [
        args.output_planner_file,
        args.output_validator_select_file,
        args.output_validator_condition_file,
        args.output_validator_join_file,
        args.output_validator_order_file,
        args.output_fixed_sql_file,
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    EOS_TOKEN = '<|eot_id|>'

    # Load input samples
    input_samples = []
    with open(args.input_file) as fp:
        for line in fp:
            input_samples.append(json.loads(line))
            
    print(f"Loaded {len(input_samples)} samples from {args.input_file}")

    # Determine the number of already processed samples by checking output files
    prompt_planner_inputs = [x['prompt_planner'] for x in input_samples]
    
    # processed_questions = {}
    # if os.path.exists(args.output_planner_file):
    #     with open(args.output_planner_file) as fp:
    #         for line in fp:
    #             line_sample = json.loads(line)
    #             for sample in line_sample:
    #                 key = f"{sample['db_id']} {sample['question']}"
    #                 processed_questions[key] = True
    #     input_samples = [sample for sample in input_samples if f"{sample['db_id']} {sample['question']}" not in processed_questions]

    if os.path.exists(args.output_planner_file):
        with open(args.output_planner_file) as fp:
            processed_count = sum(1 for _ in fp)
        input_samples = input_samples[processed_count:]

    # Load GPT planner data
    
    if args.gpt_planner_file is not None:
        gpt_question2planner = {}
        with open(args.gpt_planner_file) as fp:
            for line in fp:
                data = json.loads(line)
                gpt_question2planner[data['question']] = data['planner_combine_with_true_sql']
                if type(gpt_question2planner[data['question']]) == list:
                    gpt_question2planner[data['question']] = gpt_question2planner[data['question']][0]
    else:
        gpt_question2planner = None

    # Define a wrapper function for `process_sample` to pass additional data
    def process_sample_wrapper(sample):
        try:
            result = process_sample(sample)  # Replace with your actual processing function
            return result, None
        except Exception as e:
            error_message = f"Error processing sample {sample}: {e}\n{traceback.format_exc()}"
            return None, error_message

    # Process samples in parallel and write to output files immediately after processing each sample
    with open(args.output_planner_file, 'a+') as output_planner_fp, \
         open(args.output_validator_select_file, 'a+') as output_validator_select_fp, \
         open(args.output_validator_condition_file, 'a+') as output_validator_condition_fp, \
         open(args.output_validator_join_file, 'a+') as output_validator_join_fp, \
         open(args.output_validator_order_file, 'a+') as output_validator_order_fp, \
         open(args.output_fixed_sql_file, 'a+') as output_fixed_sql_fp:

        with Pool(processes=32) as pool:
            for result, error in tqdm(pool.imap_unordered(process_sample_wrapper, input_samples), total=len(input_samples)):
                if error:
                    print(error)  # Print the error message if there is one
                    continue  # Skip to the next sample if an error occurred
            # for result, error in tqdm(process_sample_wrapper(sample) for sample in input_samples):

                # Write each result to the corr esponding file as it's processed
                output_planner_fp.write(json.dumps(result['planner']) + '\n')
                output_validator_select_fp.write(json.dumps(result['validator_select']) + '\n')
                output_validator_condition_fp.write(json.dumps(result['validator_condition']) + '\n')
                output_validator_join_fp.write(json.dumps(result['validator_join']) + '\n')
                output_validator_order_fp.write(json.dumps(result['validator_order']) + '\n')
                output_fixed_sql_fp.write(json.dumps(result['fixed_sql']) + '\n')

    # Create Hugging Face datasets and save them
    def make_and_save_hf_dataset(filepath):
        dataset = make_hf_dataset(filepath)
        print(dataset)
        # if len dataset is empty, then skip saving
        if len(dataset['train_dpo']) == 0:
            print(f"Dataset is empty. Skipping saving to disk.")
            return dataset
        output_dataset_dir = filepath.replace(".jsonl", "")
        dataset.save_to_disk(output_dataset_dir)
        print(f"Dataset saved at: {output_dataset_dir}")
        return dataset
    
    def make_and_save_hf_dataset_from_dpo_ranking_file(filepath):
        samples = []
        added_samples = set()
        with open(filepath) as fp:
            for line in fp:
                line_sample = json.loads(line)
                for sample in line_sample:
                    prompt = sample['prompt']
                    chosen = sample['chosen']
                    rejected = sample['rejected']
                    sample = {
                        'prompt': sample['prompt'],
                        'chosen': sample['chosen'],
                        'rejected': sample['rejected']
                    }
                    key = f"{prompt} {chosen} {rejected}".strip()
                    if key not in added_samples:
                        samples.append(sample)

        dataset = DatasetDict({
            'train_dpo': Dataset.from_list(samples),
            'test_dpo': Dataset.from_list(samples[:100])
        })
        print(dataset)
        # if len dataset is empty, then skip saving
        if len(dataset['train_dpo']) == 0:
            print(f"Dataset is empty. Skipping saving to disk.")
            return dataset
        output_dataset_dir = filepath.replace(".jsonl", "")
        dataset.save_to_disk(output_dataset_dir)
        print(f"Dataset saved at: {output_dataset_dir}")
        return dataset

    print("planner dataset")
    # planner_dataset = make_and_save_hf_dataset(args.output_planner_file)
    planner_dataset = make_and_save_hf_dataset_from_dpo_ranking_file(args.output_planner_file)
    print("validator select dataset")
    validator_select_dataset = make_and_save_hf_dataset(args.output_validator_select_file)
    print("validator condition dataset")
    validator_condition_dataset = make_and_save_hf_dataset(args.output_validator_condition_file)
    print("validator join dataset")
    validator_join_dataset = make_and_save_hf_dataset(args.output_validator_join_file)
    print("validator order dataset")
    validator_order_dataset = make_and_save_hf_dataset(args.output_validator_order_file)
    print("fixed sql dataset")
    fixed_sql_dataset = make_and_save_hf_dataset(args.output_fixed_sql_file)

    print("Done!")
