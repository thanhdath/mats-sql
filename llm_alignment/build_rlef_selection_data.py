import json
import pickle as pkl
import argparse
import numpy as np
import requests
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from data_processing.planner import SelectionAgentWithSchema
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str, default='logs/results-orpo-iter-2-bird-train-top-20-temperature-1.0.pkl')
parser.add_argument("--max_candidates", type=int, default=3)
parser.add_argument("--progress_file", type=str, default='temp/bird_selection_dpo.jsonl')
args = parser.parse_args()

# Initialize Selection Agent
selection_agent = SelectionAgentWithSchema()

def get_answer_selection(messages):
    response = requests.post(
        "http://192.168.1.108:8006/v1/completions",
        json={
            "model": 'selection',
            "prompt": messages[0]['content'],
            "max_tokens": 512,
            "use_beam_search": False,
            "n": 20,
            "temperature": 1.0,
            "stop": ['<|eot_id|>', '<|end|>', '<|end_header_id|>', '<|end_of_text|>', '<｜end▁of▁sentence｜>']
        }
    ).json()
    
    try:
        return [x['text'] for x in response['choices']]
    except:
        print(response)
        return []

selection_agent.get_answer = get_answer_selection

# Load predictions
preds = pkl.load(open(args.pred_file, 'rb'))

# Load progress from previous runs
processed_keys = {}
if os.path.exists(args.progress_file):
    with open(args.progress_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            key = (sample["db_id"], sample["question"])
            processed_keys[key] = processed_keys.get(key, 0) + 1

# Expand preds 4 times and filter already processed ones
all_preds = preds * 4
filtered_preds = []
for sample in all_preds:
    key = (sample["db_id"], sample["question"])
    if processed_keys.get(key, 0) < 4:
        filtered_preds.append(sample)
        processed_keys[key] = processed_keys.get(key, 0) + 1  # Track count

def build_dpo_data(sample):
    """Process a single sample and return DPO data."""
    sample = deepcopy(sample)

    # Filter out samples with execution failures
    valid_sqls, valid_results, valid_corrects = [], [], []
    for i in range(min(len(sample['predict_sqls']), 20)):
        if 'Execution failed' not in sample['pred_results'][i] and 'too much time' not in sample['pred_results'][i]:
            valid_sqls.append(sample['predict_sqls'][i])
            valid_results.append(sample['pred_results'][i])
            valid_corrects.append(sample['is_execution_corrects'][i])

    sample['predict_sqls'] = valid_sqls
    sample['pred_results'] = valid_results
    sample['is_execution_corrects'] = valid_corrects

    # Shuffle valid results
    indices = np.random.permutation(len(sample['predict_sqls'])).tolist()
    sample['predict_sqls'] = [sample['predict_sqls'][i] for i in indices]
    sample['pred_results'] = [sample['pred_results'][i] for i in indices]
    sample['is_execution_corrects'] = [sample['is_execution_corrects'][i] for i in indices]

    # Select a random number of candidates
    n_candidates = np.random.randint(2, 6)
    sample['predict_sqls'] = sample['predict_sqls'][:n_candidates]
    sample['pred_results'] = sample['pred_results'][:n_candidates]
    sample['is_execution_corrects'] = sample['is_execution_corrects'][:n_candidates]
    sample['candidate_sqls'] = sample['predict_sqls']
    sample['candidate_pred_results'] = sample['pred_results']

    # Generate prompt and answers
    prompt, answers = selection_agent.generate(sample)

    dpo_data = {
        'db_path': sample['db_path'],
        'db_id': sample['db_id'],
        'question': sample['question'],
        'sql': sample['sql'],
        'true_result': str(sample['true_result']).strip(),
        'predict_sqls': sample['predict_sqls'],
        'pred_results': [str(x).strip() for x in sample['pred_results']],
        'is_execution_corrects': sample['is_execution_corrects'],
        'reward_data': []
    }

    for answer in answers:
        answer_index = selection_agent.extract_answer_index(answer)

        if answer_index == -1 and sum(sample['is_execution_corrects']) > 0:
            reward = 0
        elif answer_index == -1 and sum(sample['is_execution_corrects']) == 0:
            reward = 1
        elif answer_index > len(sample['is_execution_corrects']):
            reward = 0
        elif answer_index > 0:
            reward = int(sample['is_execution_corrects'][answer_index - 1])
        else:
            reward = -2

        dpo_data['reward_data'].append({
            'prompt': prompt,
            'completion': answer,
            'reward': reward
        })

    return dpo_data

if __name__ == "__main__":
    num_processes = min(32, cpu_count())  # Use up to 32 processes

    # Track progress and write every 50 samples
    processed_count = 0
    with Pool(num_processes) as pool, open(args.progress_file, 'a', encoding='utf-8') as f:
        for dpo_data in tqdm(pool.imap_unordered(build_dpo_data, filtered_preds), total=len(filtered_preds)):
            f.write(json.dumps(dpo_data, ensure_ascii=False) + "\n")
            processed_count += 1
            
            # Save every 50 samples
            if processed_count % 50 == 0:
                f.flush()
