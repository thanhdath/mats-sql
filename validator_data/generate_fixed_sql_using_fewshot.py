import json
import sqlite3
import os
import multiprocessing.pool
import functools
from tqdm import tqdm
import pandas as pd
from utils import get_columns_in_select_clause

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator

@timeout(30)
def _execute_sql_with_timeout(db_path, action):
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    actions = action.split(";")
    actions = [x for x in actions if len(x.strip()) > 0]
    if len(actions) == 0:
        return "no SQL query executed.", True
    cursor = conn.cursor()
    for action in actions:
        # action = action.lower()
        try:
            cursor.execute(action)
            response = cursor.fetchall()
            has_error = False
        except Exception as error:
            # If the SQL query is invalid, return error message from sqlite
            response = str(error)
            has_error = True
            cursor.close()
            break
    cursor.close()
    conn.close()
    return response, has_error
    
def _execute_sql(db_path, sql_query):
    try:
        pred_result, has_error = _execute_sql_with_timeout(db_path, sql_query)
    except:
        pred_result = "The query takes too much time."
        has_error = True
    return pred_result, has_error

def _make_str_response(response, has_error):
    if has_error:
        return str(response)
    else:     
        df = pd.DataFrame(response)
        return str(df)
    
# PROMPT = open('./few_shot_prompt_fix.txt').read() + """=========
# {schema}

# Matched contents are written in this format table.column (some values can be found in that column)
# {matched_content}

# Question: {question}

# SQL query: {sql_query}

# Execution response [written in pandas format]:
# {execution_response}

# Feedback:{feedback}

# FIXED SQL:"""

PROMPT = open('./few_shot_prompt_fix.txt').read().strip() + """
=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Feedback:{feedback}

FIXED SQL:"""


from openai import OpenAI

client = OpenAI(
    api_key='no-key',
    base_url='http://localhost:8000/v1'
)

# def get_answer(messages):
#     response = client.chat.completions.create(
#         model='codeS',
#         messages=messages,
#         max_tokens=2048,
#         temperature=0.0,
#         # eos_token_id=self.tokenizer.convert_tokens_to_ids(['<|end|>'])
#     )
#     response = response.choices[0].message.content.strip()
#     return response

# def get_answer(messages):
#     response = client.completions.create(
#         model='meta-llama/Meta-Llama-3.1-8B-Instruct/',
#         prompt=messages[0]['content'],
#         max_tokens=256,
#         temperature=0.0,
#         stop=['=========']
#         # eos_token_id=self.tokenizer.convert_tokens_to_ids(['<|end|>'])
#     )
#     response = response.choices[0].text
#     return response

def get_answer(messages):
    import requests
    response = requests.post("http://localhost:8000/v1/completions",
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct/",
                "prompt": messages[0]['content'],
                "max_tokens": 256,
                "use_beam_search": True,
                "n": 4,
                "temperature": 0,
                "stop": ["========="]
                }).json()
    return response["choices"][0]["text"]

data = json.load(open('./bird_validator_select.json'))
output_file = './bird_fixed_sql.json'

# data = json.load(open('../temp/codes/temp/codes/eval_codes-1b.json'))
# output_file = 'bird_dev_validator_select.json'

for isample in tqdm(range(0, len(data)), total=len(data)):
    sample = data[isample]

    sql = sample['predict_sql']
    is_correct = sample['is_correct']
    if sample['validator_select'] is None or "Conclude: correct" in sample['validator_select']:
        continue

    prompt = PROMPT.format(
        schema=sample['schema_sequence'], 
        matched_content=sample['content_sequence'],
        question=sample['text'],
        sql_query=sql,
        # execution_response=sample['pred_result'],
        feedback=sample['validator_select']
    )
    # print(prompt)
    answer = get_answer([{"role": "user", "content": prompt}])

    execution_result = _execute_sql("../" + sample['db_path'], answer)

    print("-"*20)
    print(answer)
    # break
    sample['fixed_sql'] = answer
    sample['fixed_pred_result'] = _make_str_response(*execution_result)

    json.dump(data[:isample+1], open(output_file, 'w+'), ensure_ascii=False, indent=4)
json.dump(data[:isample+1], open(output_file, 'w+'), ensure_ascii=False, indent=4)

bird_results_dict = dict()
for idx, sample in enumerate(data):
    if 'fixed_sql' in sample:
        predicted_sql = sample['fixed_sql']
    else:
        predicted_sql = sample['predict_sql']
    bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + sample["db_id"]
with open("predict_dev.json", "w", encoding = 'utf-8') as f:
    f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))
