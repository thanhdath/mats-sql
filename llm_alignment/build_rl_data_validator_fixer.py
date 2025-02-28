import json
from tqdm import tqdm
from validator_data.validator import _execute_sql, execute_sql_with_time, get_answer_openai
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
        execution_result = data['execution_result']
        if 'too much time' in execution_result: # ignore the sample that takes too much time
            continue

        if prompt not in prompt_to_completions:
            prompt_to_completions[prompt] = []

        completion = '\n' + completion.strip()
        prompt_to_completions[prompt].append({
            'completion': completion,
            'reward': reward
        })

    dpo_data = [] # each element is a dictionary with keys: prompt, chosen, rejected
    for prompt, completions in prompt_to_completions.items():

        completions = sorted(completions, key=lambda x: x['reward'], reverse=True)
        reward_1s = [x for x in completions if x['reward'] == 1]
        reward_0s = [x for x in completions if x['reward'] == 0]
        reward_ns = [x for x in completions if x['reward'] == -1]

        chosens = reward_1s
        rejecteds = reward_ns + reward_0s

        np.random.shuffle(chosens)
        np.random.shuffle(rejecteds)
        # Ensure we have at least one rejected entry to form pairs
        n_select = args.n_select_chosens
        if len(rejecteds) < n_select:
            rejecteds += [{
                'completion': "",
                'reward': -2
            }] * (n_select - len(rejecteds))

        assert len(rejecteds) >= n_select

        # Form pairs by combining `chosens` and `rejecteds`
        # for chosen, rejected in zip(chosens[:n_select], rejecteds[:n_select]):
        #     dpo_sample = {
        #         'prompt': prompt,
        #         'chosen': chosen['completion'],
        #         'rejected': rejected['completion'],
        #     }
        #     dpo_data.append(dpo_sample)

        for chosen in chosens[:5]:
            for rejected in rejecteds[:5]:
                dpo_sample = {
                    'prompt': prompt,
                    'chosen': chosen['completion'],
                    'rejected': rejected['completion'],
                }
                dpo_data.append(dpo_sample)

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
        return None, None, None
    return execute_sql_with_time("./" + db_path, sql)

def is_the_same_sql(sql1, sql2, db_path):
    # norm \s+ to " " and strip
    # sql1 = re.sub(r'\s+', ' ', sql1).strip().lower()
    # sql2 = re.sub(r'\s+', ' ', sql2).strip().lower()
    # # remove ``
    # sql1 = sql1.replace('`', '')
    # sql2 = sql2.replace('`', '')

    # execute and check if is_execution_correct
    sql1 = execute_sql_with_time("./" + db_path, sql1)[0]
    sql2 = execute_sql_with_time("./" + db_path, sql2)[0]
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

    new_execution = execute_sql_with_time("./" + sample['db_path'], fixed_sql)

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
    true_execution_result = execute_sql_with_time("./" + sample['db_path'], sample['sql'])
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

    # Initialize data lists
    planner_data = {
        'db_path': sample['db_path'],
        'db_id': sample['db_id'],
        'question': sample['question'],
        'sql': sample['sql'],
        'true_execution_result': str(true_execution_result[0]),
        'true_execution_time': true_execution_result[2],
        'reward_data': []
    }
    fixed_sql_data = {
        'db_path': sample['db_path'],
        'db_id': sample['db_id'],
        'question': sample['question'],
        'sql': sample['sql'],
        'true_execution_result': str(true_execution_result[0]),
        'true_execution_time': true_execution_result[2],
        'reward_data': []
    }

    # Process GPT planner and check correctness
    if gpt_question2planner is not None:
        if sample['question'] in gpt_question2planner:
            gpt_plan = gpt_question2planner[sample['question']]
            gpt_predict_sql = get_predict_sql_from_planner(gpt_plan)
            gpt_execution_result = execute_sql_with_time("./" + sample['db_path'], gpt_predict_sql)

            completion = gpt_question2planner[sample['question']]
            #completion = concat_db_response_to_completion(completion, gpt_execution_result)

            if is_execution_correct(true_execution_result[0], gpt_execution_result[0]):
                planner_data['reward_data'].append({
                    'prompt': sample['prompt_planner'][0],
                    'completion': completion,
                    'reward': 1,
                    'execution_result': str(gpt_execution_result[0]),
                    'execution_time': gpt_execution_result[2]
                })


    # Process fixed SQLs if present
    if 'prompt_fix' in sample:
        prompt_fixs = sample['prompt_fix'][:1]

        # if args.enable_advanced_fix_agent:
        #     gpt_fixed_sqls = [process_prompt_fix(prompt_fix) for prompt_fix in prompt_fixs]

        #     # prompt2sql = open('prompt2sql.json').read()
        #     # prompt2sql = json.loads(prompt2sql)
        #     # gpt_fixed_sqls = [prompt2sql[prompt_fix] if prompt_fix in prompt2sql else None for prompt_fix in prompt_fixs]
            
        #     # Use multi-threading for fixed SQL execution
        #     fixed_sql_args = [(sample['db_path'], sql) for sql in gpt_fixed_sqls]
        #     with ThreadPoolExecutor(max_workers=32) as executor:
        #         gpt_execution_results = list(executor.map(execute_sql_with_path, fixed_sql_args))
            
        #     for prompt_fix, gpt_fixed_sql, exec_result in zip(prompt_fixs, gpt_fixed_sqls, gpt_execution_results):
        #         if gpt_fixed_sql is not None:
        #             fixed_sql_data.append({
        #                 'prompt': prompt_fix,
        #                 'completion': gpt_fixed_sql,
        #                 'reward': int(is_execution_correct(true_execution_result[0], exec_result[0])),
        #                 'db_id': sample['db_id'],
        #                 'question': sample['question'],
        #                 'db_path': sample['db_path'],
        #                 'sql': sample['sql']
        #             })

    for i, (plan, is_correct, execution_result) in enumerate(zip(sample['planners'], is_planner_sql_execution_corrects, planner_sql_execution_results)):
        pred_response, pred_sql_has_error, pred_exec_time = execution_result
        if pred_sql_has_error:
            planner_data['reward_data'].append({
                'prompt': sample['prompt_planner'][i],
                'completion': plan,
                'reward': -1,
                'execution_result': str(pred_response),
                'execution_time': pred_exec_time
            })
        else:
            planner_data['reward_data'].append({
                'prompt': sample['prompt_planner'][i],
                'completion': plan,
                'reward': int(is_correct),
                'execution_result': str(pred_response),
                'execution_time': pred_exec_time
            }) 

    # Append data for each predicted SQL
    for i, (fixed_sql, is_correct, planner_sql, execution_result) in enumerate(zip(fixed_sqls, is_fixed_sql_execution_corrects, planner_sqls, fixed_sql_execution_results)):
        if fixed_sql is not None:
            fixed_sql_data['reward_data'].append({
                'prompt': sample['prompt_fix'][i],
                'completion': sample['feedbacks'][i],
                'reward': 0 if is_the_same_sql(fixed_sql, planner_sql, sample['db_path']) else int(is_correct),
                'execution_result': str(execution_result[0]),
                'execution_time': execution_result[2]
            })

    # Filter out samples with None values for prompt or completion
    planner_data['reward_data'] = [x for x in planner_data['reward_data'] if x['prompt'] is not None and x['completion'] is not None]
    fixed_sql_data['reward_data'] = [x for x in fixed_sql_data['reward_data'] if x['prompt'] is not None and x['completion'] is not None]

    return {
        'planner': planner_data,
        'fixed_sql': fixed_sql_data
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
    parser.add_argument("--output_fixed_sql_file", default="data/llm_alignment/p5/dpo-llama-3-end2end_spider_train_fixed_sql.jsonl", help="Path for the fixed SQL output JSONL file")

    parser.add_argument("--enable_advanced_fix_agent", action="store_true", help="Enable the GPT fix agent")
    parser.add_argument("--n_select_chosens", default=2, type=int)
    args = parser.parse_args()

    # Ensure output directories exist
    for path in [
        args.output_planner_file,
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
    
    processed_questions = {}
    if os.path.exists(args.output_planner_file):
        with open(args.output_planner_file) as fp:
            for line in fp:
                sample = json.loads(line)
                key = f"{sample['db_id']} {sample['question']}"
                processed_questions[key] = True
        input_samples = [sample for sample in input_samples if f"{sample['db_id']} {sample['question']}" not in processed_questions][::-1]
    
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
    # with open(args.output_planner_file, 'a+') as output_planner_fp, \
    #      open(args.output_fixed_sql_file, 'a+') as output_fixed_sql_fp:

    #     with Pool(processes=4) as pool:
    #         for result, error in tqdm(pool.imap_unordered(process_sample_wrapper, input_samples), total=len(input_samples)):
    #             if error:
    #                 print(error)  # Print the error message if there is one
    #                 continue  # Skip to the next sample if an error occurred
    #             # Write each result to the corr esponding file as it's processed
    #             output_planner_fp.write(json.dumps(result['planner']) + '\n')
    #             output_fixed_sql_fp.write(json.dumps(result['fixed_sql']) + '\n')

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
                all_rewards_data = json.loads(line)['reward_data']
                dpo_data = build_dpo_ranking_data(all_rewards_data)

                for sample in dpo_data:
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
    planner_dataset = make_and_save_hf_dataset_from_dpo_ranking_file(args.output_planner_file)
    print("fixed sql dataset")
    # fixed_sql_dataset = make_and_save_hf_dataset(args.output_fixed_sql_file)
    fixed_sql_dataset = make_and_save_hf_dataset_from_dpo_ranking_file(args.output_fixed_sql_file)

    print("Done!")
