import json
import sqlite3
import os
import multiprocessing.pool
import functools
from tqdm import tqdm
import pandas as pd
from validator import FixAgent, _make_str_response, _execute_sql, is_execution_correct

PROMPT = open('./few_shot_prompt_fix_join.txt').read().strip() + """
=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Feedback:{feedback}

FIXED SQL:"""

data = []
with open('../data/llm_alignment/validator_condition_bird_with_evidence_dev.jsonl') as fp:
    for line in fp:
        data.append(json.loads(line))

output_file = './bird_fixed_sql_condition.jsonl'

output_fp = open(output_file, 'w+')

fix_agent = FixAgent(PROMPT, endpoint_type='vllm')

for isample in tqdm(range(len(data)), total=len(data)):
    sample = data[isample]

    is_correct = sample['is_correct']
    if sample['validator_condition'] is None or "Conclude: correct" in sample['validator_condition']:
        output_fp.write(json.dumps(sample) + '\n')
        continue

    prompt = PROMPT.format(
        schema=sample['schema_sequence'], 
        matched_content=sample['content_sequence'],
        question=sample['text'],
        sql_query=sample['predict_sql'],
        # execution_response=sample['pred_result'],
        feedback=sample['validator_condition']
    )
    # print(prompt)
    answer = fix_agent.get_answer([{"role": "user", "content": prompt}])
    execution_result = _execute_sql("../" + sample['db_path'], answer)

    print("-"*20)
    print(answer)
    # break
    sample['fixed_sql'] = answer
    sample['fixed_pred_result'] = _make_str_response(*execution_result)

    true_execution_result = _execute_sql("../" + sample['db_path'], sample['sql'])
    is_fixed_correct = is_execution_correct(true_execution_result[0], execution_result[0])
    sample['is_fixed_correct'] = is_fixed_correct

    output_fp.write(json.dumps(sample) + '\n')

bird_results_dict = dict()
for idx, sample in enumerate(data):
    if 'fixed_sql' in sample:
        predicted_sql = sample['fixed_sql']
    else:
        predicted_sql = sample['predict_sql']
    bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + sample["db_id"]
with open("predict_dev.json", "w", encoding = 'utf-8') as f:
    f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))
output_fp.close()

