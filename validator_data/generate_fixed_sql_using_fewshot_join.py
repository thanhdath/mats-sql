import json
from tqdm import tqdm
import pandas as pd
import argparse
from validator import FixAgent, _execute_sql, _make_str_response, is_execution_correct

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../data/llm_alignment/validator_join_bird_with_evidence_train.jsonl')
parser.add_argument('--output_file', type=str, default='../data/llm_alignment/fixed_join_bird_with_evidence_train.jsonl')
parser.add_argument('--endpoint_type', type=str, default='llamacpp', choices=['vllm', 'llamacpp'])
args = parser.parse_args()

PROMPT = open('./few_shot_prompt_fix_join.txt').read().strip() + """
=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Feedback:{feedback}

FIXED SQL:"""

# load data from jsonl
data = []
with open(args.input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

fix_agent = FixAgent(prompt_template=PROMPT, endpoint_type=args.endpoint_type)

output_file = open(args.output_file, 'a+')

for isample in tqdm(range(0, len(data)), total=len(data)):
    sample = data[isample]

    sql = sample['predict_sql']
    is_correct = sample['is_correct']
    if sample['validator_join'] is None or "Conclude: correct" in sample['validator_join']:
        output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
        continue

    prompt = PROMPT.format(
        schema=sample['schema_sequence'], 
        matched_content=sample['content_sequence'],
        question=sample['text'],
        sql_query=sql,
        feedback=sample['validator_join']
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

    output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

bird_results_dict = dict()
for idx, sample in enumerate(data):
    if 'fixed_sql' in sample:
        predicted_sql = sample['fixed_sql']
    else:
        predicted_sql = sample['predict_sql']
    bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + sample["db_id"]
with open("predict_dev.json", "w", encoding = 'utf-8') as f:
    f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))
