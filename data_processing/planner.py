
import sqlite3
import multiprocessing.pool
import functools
import pandas as pd
import re
import sqlparse
import requests
from sql_metadata import Parser
from data_processing.utils import get_table_columns_list, remove_table_alias, get_columns_in_select_clause, get_equation_function_in_select_clause, remove_table_alias
from openai import OpenAI
import os
from dotenv import load_dotenv
from func_timeout import func_set_timeout, FunctionTimedOut
from copy import deepcopy

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 10)

# execute predicted sql with a time limitation
@func_set_timeout(30)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()


def check_sql_executability(generated_sql, db):
    if not os.path.exists(db):
        raise Exception("Database file not found: %s" % db)
        
    connection = sqlite3.connect(db, check_same_thread = False)
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()

    if generated_sql.strip() == "":
        return "Error: empty string"
    try:
        execute_sql(cursor, "EXPLAIN QUERY PLAN " + generated_sql)
        execution_error = None
    except FunctionTimedOut as fto:
        print("SQL execution time out error: {}.".format(fto))
        execution_error = "SQL execution times out."
    except Exception as e:
        # print("SQL execution runtime error: {}.".format(e))
        execution_error = str(e)

    cursor.close()
    connection.close()
    
    return execution_error

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
            # cursor.execute(action)
            # response = cursor.fetchall()
            response = pd.read_sql_query(action, conn)
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
        # df = pd.DataFrame(response)
        # return str(df)
        return str(response).strip()

def is_execution_correct(true_response, pred_response):
    """
    Return True if both true_response and pred_response are pandas DataFrames
    and they have the same rows (ignoring row order), otherwise False.
    A string response is treated as an execution error, so it returns False.
    """
    
    # If either response is a string, treat it as an error => not correct.
    if isinstance(true_response, str) or isinstance(pred_response, str):
        return False
    
    # Fill missing values in both DataFrames to avoid NaN comparison issues.
    true_response = true_response.fillna("")
    pred_response = pred_response.fillna("")
    
    # Convert each row of the DataFrame into a tuple, then compare sets.
    # This makes ordering irrelevant (only set membership matters).
    true_set = set(map(tuple, true_response.values.tolist()))
    pred_set = set(map(tuple, pred_response.values.tolist()))
    
    return true_set == pred_set


# def get_answer(messages):
#     response = client.completions.create(
#         model='meta-llama/Meta-Llama-3.1-8B-Instruct/',
#         prompt=messages[0]['content'],
#         max_tokens=256,
#         temperature=0.0,
#         use_beam_search=True,
#         n=4,
#         stop=['=========']
#         # eos_token_id=self.tokenizer.convert_tokens_to_ids(['<|end|>'])
#     )
#     response = response.choices[0].text
#     return response
    
def get_answer_vllm(messages):
    response = requests.post(
        # "http://localhost:8000/v1/completions",
        "http://192.168.1.117:8000/v1/completions",
            json={
                # "model": "meta-llama/Meta-Llama-3.1-8B-Instruct/",
                "model": "/hdd/datht/huggingface/qwen-1b-bird-planner/",
                "prompt": messages[0]['content'],
                "max_tokens": 768,
                "use_beam_search": True,
                "n": 4,
                "temperature": 0,
                "stop": ["========="]
                }).json()
    # select a choice that has text
    choices = [choice for choice in response["choices"] if choice["text"]]
    if len(choices) > 0:
        return choices[0]['text']
    else:
        return response["choices"][0]['text']


def get_answer_llamacpp(messages):
    response = requests.post("http://localhost:8000/v1/completions",
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct/",
                "prompt": messages[0]['content'],
                "n_predict": 768,
                "temperature": 0,
                "stop": ["========="]
                }).json()
    return response["content"]

def get_answer_openai(client, messages, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=768,
        temperature=0.0,
    )
    response = response.choices[0].message.content.strip()
    return [response]
    

class Planner:
    def __init__(self, prompt_file, endpoint_type='llamacpp'):
        load_dotenv()

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm
        elif endpoint_type == 'openai':
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.get_answer = lambda x: get_answer_openai(client, x)

        self.prompt_template = open(prompt_file).read() + """
=========
{schema}

Question: {question}
External knowledge: {evidence}

"""

    def generate(self, sample):
        if 'prompt' not in sample:
            prompt = self.prompt_template.format(
                schema=sample['schema_sequence'], 
                question=sample['question'],
                evidence=sample['evidence']
            )
        else:
            prompt = sample['prompt']

        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers
    
class PlannerCombine(Planner):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(prompt_file='./prompts/few_shot_prompt_planner_combine.txt', endpoint_type=endpoint_type)

class PlannerCombineWithTrueSQLRefiner(Planner):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(prompt_file='./prompts/few_shot_prompt_planner_combine.txt', endpoint_type=endpoint_type)

#         self.prompt_template = open('./prompts/few_shot_prompt_planner_combine.txt').read() + """
# =========
# {schema}

# Matched contents are written in this format table.column (some values can be found in that column)
# {matched_content}

# Question: {question}

# Use this hidden True SQL query to write correct analysis that derives to the correct answer. The True SQL query cannot be used in the analysis.
# Hidden True SQL query: {true_sql_query} 

# Answer like example format:"""
        self.prompt_template = """{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

Use this hidden True SQL query to write correct analysis that derives to the correct answer. The True SQL query cannot be used in the analysis.
Hidden True SQL query: {true_sql_query} 

Answer like example format:"""

    def generate(self, sample):
        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            matched_content=sample['content_sequence'],
            question=sample['text'],
            true_sql_query=sample['sql']
        )
        answer = sample['planner_combine_with_true_sql']
        messages = [{"role": "user", "content": prompt}]
        messages.append({"role": "assistant", "content": answer})
#         # add prompt to refine the answer
        messages.append({
            'role': 'user',
            'content': f"""The true SQL query returns this result:
{sample['true_result']}
The predicted SQL query returns this result:
{sample['pred_result']}

Please rewrite the plan to generate the correct answer. The answer format must the same as the example format above. The final SQL query must be the same as the hidden True SQL query.
Add additional thoughts after Tables to use and before Final SQL query if needed. Do not mention about the previous plan or previous SQL. The select goal must be the same as the True SQL query.
Answer in the example format:"""})

        answer = self.get_answer(messages)
        return answer

class PlannerCombineWithTrueSQL(Planner):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(prompt_file='./data_processing/prompts/few_shot_prompt_planner_combine.txt', endpoint_type=endpoint_type)

        self.prompt_template = open('./data_processing/prompts/few_shot_prompt_planner_combine.txt').read() + """
=========
{schema}

Question: {question}
External knowledge: {evidence}

Use this hidden True SQL query to write correct analysis that derives to the correct answer. The True SQL query cannot be used in the analysis.
Hidden True SQL query: {true_sql_query} 

Always use external knowledge if it has been provided. Known that the database is SQLite.
Answer like example format:"""

    def generate(self, sample):
        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            # matched_content=sample['content_sequence'],
            question=sample['question'],
            evidence=sample.get('evidence', 'None'),
            true_sql_query=sample['sql']
        )
        answer = self.get_answer([{"role": "user", "content": prompt}])
        return answer

class ValidatorJOIN:
    def __init__(self, endpoint_type='llamacpp'):
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.max_columns', 10)

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm

        self.prompt_template = open('./few_shot_prompt_join.txt').read() + """=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Feedback:
JOIN.
- The SQL query uses tables {used_tables}, joining them on foreign keys {used_fks}."""

    def get_table_list(self, schema):
        tables = []
        for table_data in schema['schema_items']:
            table_name = table_data['table_name'].lower()
            tables.append(table_name)
        tables = list(set(tables))
        return tables
    
    def extract_join_clause(self, sql_query):
        # Define a regex pattern to match the SELECT clause up to the FROM keyword
        pattern = re.compile(r"FROM\s.*?\s(?=WHERE)", re.IGNORECASE | re.DOTALL)
        
        # Search for the pattern in the SQL query
        match = pattern.search(sql_query)
        
        if match:
            # Return the matched portion (SELECT clause)
            return match.group(0).strip()
        else:
            pattern  = re.compile(r"FROM.+", re.IGNORECASE | re.DOTALL)
            # Return None if no match is found
            # Search for the pattern in the SQL query
            match = pattern.search(sql_query)
            
            if match:
                # Return the matched portion (SELECT clause)
                return match.group(0).strip()
            else:
                return None

    def get_used_fks(self, sql_query):
        # use re, get all condition join after ON
        pattern = re.compile(r" ON\s.*?(?=\sWHERE|\sORDER BY|\sLIMIT|\sGROUP BY)", re.IGNORECASE | re.DOTALL)
        match = pattern.findall(sql_query)
        return match
 

    def get_tables_in_join_clause(self, sql_query, schema):
        table_list = self.get_table_list(schema)
        sql_query = remove_table_alias(sqlparse.format(sql_query, keyword_case = "upper", identifier_case = "lower"))
        join_clause = self.extract_join_clause(sql_query)

        used_tables = []
        for token in join_clause.split():
            if token in table_list:
                used_tables.append(token)

        used_fks = self.get_used_fks(sql_query)
        return used_tables, used_fks

    def validate(self, sample):
        execution_result = _execute_sql("../" + sample['db_path'], sample['predict_sql'])
        used_tables, used_fks = self.get_tables_in_join_clause(sample['predict_sql'], sample['schema'])

        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            matched_content=sample['content_sequence'],
            question=sample['text'],
            sql_query=sample['predict_sql'],
            execution_response=_make_str_response(*execution_result),
            used_tables=used_tables,
            used_fks=used_fks
        )
        answer = prompt.split("Feedback:")[-1] + self.get_answer([{"role": "user", "content": prompt}])
        return answer, execution_result
    
class ValidatorOrder:
    def __init__(self, endpoint_type='llamacpp'):
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.max_columns', 10)

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm

        self.prompt_no_none = open('./few_shot_prompt_order.txt').read().replace("{", "{{").replace("}", "}}") + """
=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Feedback:
ORDER BY.
- The SQL query uses ```{order_by_clause}```.
- Based on the question, the query should use"""

        self.prompt_has_none = open('./few_shot_prompt_order.txt').read().replace("{", "{{").replace("}", "}}") + """
=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Feedback:
ORDER BY.
- The SQL query uses ```{order_by_clause}```.
- However, the column ```{order_by_column}```` has None values, so the SQL query need to add condition ```{order_by_column} IS NOT NULL``` to filter out None values.
- Conclude: incorrect."""

    def get_table_list(self, schema):
        tables = []
        for table_data in schema['schema_items']:
            table_name = table_data['table_name'].lower()
            tables.append(table_name)
        tables = list(set(tables))
        return tables
    
    def extract_order_clause(self, sql_tokens):
        # extract order by clause given sql_tokens is a list, find start index of order by token
        order_by_index = -1
        for i in range(len(sql_tokens)):
            if sql_tokens[i] == "order by":
                order_by_index = i
                break
        # return order clause
        if order_by_index == -1:
            return []
        else:
            return sql_tokens[order_by_index:]

    def extract_order_by_clause_using_regex(self, sql_query):
        # use regex on sql_query to extract order by clause
        order_by_clause = re.search(r'(?i)ORDER BY\s+(.*)', sql_query)
        if order_by_clause is None:
            return None
        else:
            return order_by_clause.group(1)

    def get_columns_in_order_clause(self, sql_query, schema):
        column_list = get_table_columns_list(schema)

        try:
            sql_tokens = [token.value for token in Parser(sql_query.lower()).tokens]
        except Exception as e:
            sql_tokens = sql_query.lower().split()
    
        order_clause_tokens = self.extract_order_clause(sql_tokens)

        equation_functions = []
        for token in order_clause_tokens:
            if token in ["min", "max", "avg", "sum", "count", "divide", "+", "/", "case", "when"]:
                equation_functions.append(token)

        # use regex on sql_query to extract order by clause
        order_by_clause = self.extract_order_by_clause_using_regex(sql_query)

        if len(equation_functions) > 0:
            return None, order_by_clause # not supported yet
        else:
            columns = []
            for token in order_clause_tokens:
                if token in column_list:
                    columns.append(token)

            # norm columns list, add table.column if '.' not present. table can extract using regex on sql query SELECT x FROM table
            norm_columns = []
            for column in columns:
                if "." not in column:
                    # regex find table name right after the word 'FROM', table name can be wrapped inside ``
                    table = re.search(r'(?i)FROM\s+`?(\w+)`?', sql_query).group(1)
                    norm_columns.append(f"{table}.{column}")
                else:
                    norm_columns.append(column)

            return norm_columns, order_by_clause
        
    def get_column_type(self, column, schema):
        # column is a string in form 'table.column' or 'column'
        if "." in column:
            table, column = column.split(".")
            for table_data in schema['schema_items']:
                if table_data['table_name'] == table:
                    for column_name, column_type in zip(table_data['column_names'], table_data['column_types']):
                        if column_name == column:
                            return column_type
        else:
            for table_data in schema['schema_items']:
                for column_name, column_type in zip(table_data['column_names'], table_data['column_types']):
                    if column_name == column:
                        return column_type
    
    def check_order_by_column_has_none_values(self, column, db_path):
        # use sql query to check if column has none values
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        table_name = column.split(".")[0]
        column_name = column.split(".")[1]
        query = f"SELECT COUNT(*) FROM `{table_name}` WHERE `{column_name}` IS NULL"
        try:
            c.execute(query)
            result = c.fetchall()
        except Exception as err:
            result = str(err)
        conn.close()

        if type(result) == list and result[0][0] > 0:
            return True
        else:
            return False
        
    def validate(self, sample):
        execution_result = _execute_sql("../" + sample['db_path'], sample['predict_sql'])

        order_columns, order_by_clause = self.get_columns_in_order_clause(sample['predict_sql'], sample['schema'])
        if order_columns is not None and len(order_columns) > 0:
            column = order_columns[0]
            if self.check_order_by_column_has_none_values(column, "../" + sample['db_path']) == True:
                prompt = self.prompt_has_none.format(
                    schema=sample['schema_sequence'], 
                    matched_content=sample['content_sequence'],
                    question=sample['text'],
                    sql_query=sample['predict_sql'],
                    execution_response=_make_str_response(*execution_result),
                    order_by_clause=order_by_clause,
                    order_by_column=column
                )
                answer = prompt.split("Feedback:")[-1]
                return answer, execution_result
            else: # False or error string
                prompt = self.prompt_no_none.format(
                    schema=sample['schema_sequence'], 
                    matched_content=sample['content_sequence'],
                    question=sample['text'],
                    sql_query=sample['predict_sql'],
                    execution_response=_make_str_response(*execution_result),
                    order_by_clause=order_by_clause)
                answer = prompt.split("Feedback:")[-1] + self.get_answer([{"role": "user", "content": prompt}])
        else:
            answer = None
            
        return answer, execution_result



class FixAgent():
    def __init__(self, prompt_template=None, endpoint_type='llamacpp'):

        if endpoint_type in ['openai', 'vllm']:
            self.prompt_template = """You are a SQL tutor that helps fixing the SQL query generated by a student. Given a database schema and a question with external knowledge. Generate Fixed SQL query based on the feedback. Write the SQL query directly, do not add more thoughts.

{schema}

Question: {question}
External knowledge: {evidence}

Generated SQL query from student with the execution response.
SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

The feedback for the SQL query:
{feedback_select}

{feedback_condition}

{feedback_join}

FIXED SQL:"""
        else:
            self.prompt_template = prompt_template

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            # self.get_answer = get_answer_vllm
            client = OpenAI(
                base_url="http://localhost:8003/v1",
                api_key="no-key",
            )
            self.get_answer = lambda x: get_answer_openai(client, x, model='vllm')

        elif endpoint_type == 'openai':
            from dotenv import load_dotenv
            load_dotenv()
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.get_answer = lambda x: get_answer_openai(client, x)

    def generate(self, sample, feedback_select, feedback_condition, feedback_join, feedback_order):
        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sample['predict_sql'],
            execution_response=sample['pred_result'],
            feedback_select=feedback_select,
            feedback_condition=feedback_condition,
            feedback_join=feedback_join,
        )
        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers

    def get_final_sql(self, predict_sql, fixed_sql, db_path):
        sqls_priority = [fixed_sql, predict_sql]
        sqls_priority = [sql for sql in sqls_priority if sql is not None]
        for sql in sqls_priority:
            # check if the sql is executable
            execution_error = check_sql_executability(sql, db_path)
            if execution_error is None:
                return sql
        return predict_sql
    


class SelectionAgent:
    def __init__(self, endpoint_type='llamacpp'):
        load_dotenv()

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm
        elif endpoint_type == 'openai':
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.get_answer = lambda x: get_answer_openai(client, x)

        self.prompt_template = """<|start_header_id|>user<|end_header_id|>
Given the question and following SQL queries, and execution results, please select the best SQL query that can answer the question. Answer the index of the SQL query you choose.

Question: {question}
Hint: {evidence}
"""
        self.choice_prompt = """
{index}. {sql}
Execution result: {result}
-------------------------
"""   

    def build_prompt(self, sample):
        prompt = self.prompt_template.format(question=sample['question'], evidence=sample['evidence'])
        index = 1
        for i in range(len(sample['candidate_sqls'])):
            choice_prompt = self.choice_prompt.format(index=index, sql=sample['candidate_sqls'][i].strip(), result=sample['candidate_pred_results'][i])
            index += 1

            prompt += choice_prompt
        
        prompt += """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""   
        return prompt

    def generate(self, sample):
        prompt = self.build_prompt(sample)
        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers

    def is_duplicated_execution_result(self, result, seen_results):
        # use is_execution_correct to check if the result is correct
        is_corrects = [is_execution_correct(result, x) for x in seen_results]
        return any(is_corrects)

    def get_best_sql(self, sample, max_candidates=2):
        """
        Iteratively compare predict_sqls in groups of 'max_candidates'.
        In each round, we chunk the remaining candidates into groups of size 'max_candidates'.
        For each group, we ask 'selection_agent' to pick the best SQL among them
        (1-based index) or return -1/0 to reject them all.
        
        We continue until only one candidate remains or none. Return final SQL or None.
        """

        # Extract the lists from the sample
        predict_sqls = []
        pred_results = []
        seen_results = []
        for predict_sql, pred_result in zip(sample['candidate_sqls'], sample['candidate_pred_results']):
            if 'Execution failed' not in str(pred_result) and 'too much time' not in str(pred_result) and not self.is_duplicated_execution_result(pred_result, seen_results):
                seen_results.append(pred_result)
                predict_sqls.append(re.sub(r'\s+', ' ', predict_sql).strip())
                pred_results.append(pred_result)

        compare_list = []

        while len(predict_sqls) > 1:
            new_predict_sqls = []
            new_pred_results = []

            # Split the current set of candidates into chunks of size 'max_candidates'
            for i in range(0, len(predict_sqls), max_candidates):
                chunk_sqls = predict_sqls[i : i + max_candidates]
                chunk_results = pred_results[i : i + max_candidates]

                # Build a temporary sample for just this chunk
                chunk_sample = deepcopy(sample)
                chunk_sample['candidate_sqls'] = chunk_sqls
                chunk_sample['candidate_pred_results'] = chunk_results

                # Ask the selection agent to pick the best SQL from 1..max_candidates
                # or return -1/0 to reject all
                prompt, answer_list = self.generate(chunk_sample)
                # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                # print(prompt + answer_list[0])

                if not answer_list:
                    # If no answer from the agent, reject entire chunk
                    continue

                try:
                    # Convert the agent's first answer to int
                    answer = int(answer_list[0])
                except:
                    # If parsing fails, reject entire chunk
                    answer = -1

                # add to compare list, if answer = 1, add pred_results[0] > pred_results[1]
                if len(chunk_results) == 2 and answer in [1, 2]:
                    if answer == 1:
                        # compare_list.append((chunk_results[0], chunk_results[1]))
                        compare_list.append((str(chunk_results[0]), str(chunk_results[1])))
                    elif answer == 2:
                        # compare_list.append((chunk_results[1], chunk_results[0]))
                        compare_list.append((str(chunk_results[1]), str(chunk_results[0])))

                # If agent picks 1..len(chunk_sqls), keep that candidate
                if 1 <= answer <= len(chunk_sqls):
                    chosen_idx = answer - 1
                    new_predict_sqls.append(chunk_sqls[chosen_idx])
                    new_pred_results.append(chunk_results[chosen_idx])
                else:
                    # If agent picks -1 or 0 or out-of-range -> reject entire chunk
                    pass

            # Update lists with the "winners" of each chunk
            predict_sqls = new_predict_sqls
            pred_results = new_pred_results

            # If in this round we fail to keep anything, no final single solution
            if not predict_sqls:
                # get greedy answer = first predict_sqls
                # print('Greedy answer')
                return sample['candidate_sqls'][0]

        # At the end, if there's exactly one candidate, return it; otherwise None
        if len(predict_sqls) == 1:
            return predict_sqls[0]
    
        # print('greedy answer')
        return sample['candidate_sqls'][0]



class SelectionAgentWithSchema:
    def __init__(self, endpoint_type='llamacpp'):
        load_dotenv()

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm
        elif endpoint_type == 'openai':
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.get_answer = lambda x: get_answer_openai(client, x)

        self.prompt_template = """<|start_header_id|>user<|end_header_id|>
Given the question and following SQL queries, and execution results, please select the best SQL query that can answer the question. Answer the index of the SQL query you choose.
{schema}

Question: {question}
Hint: {evidence}
"""
        self.choice_prompt = """
{index}. {sql}
Execution result: {result}
-------------------------
"""   

    def build_prompt(self, sample):
        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'],
            question=sample['question'], 
            evidence=sample['evidence'])
        index = 1
        for i in range(len(sample['candidate_sqls'])):
            choice_prompt = self.choice_prompt.format(index=index, sql=sample['candidate_sqls'][i].strip(), result=sample['candidate_pred_results'][i])
            index += 1

            prompt += choice_prompt
        
        prompt += """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""   
        return prompt

    def generate(self, sample):
        prompt = self.build_prompt(sample)
        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers

    def is_duplicated_execution_result(self, result, seen_results):
        # use is_execution_correct to check if the result is correct
        is_corrects = [is_execution_correct(result, x) for x in seen_results]
        return any(is_corrects)
    
    def extract_answer_index(self, answer):
        try:
            # Convert the agent's first answer to int
            # extract answer inside <answer> </answer>
            answer = re.search(r'<answer>(.*)</answer>', answer).group(1)
            answer = int(answer)
        except:
            # If parsing fails, reject entire chunk
            answer = -1
        return answer

    def get_best_sql(self, sample, max_candidates=2):
        """
        Iteratively compare predict_sqls in groups of 'max_candidates'.
        In each round, we chunk the remaining candidates into groups of size 'max_candidates'.
        For each group, we ask 'selection_agent' to pick the best SQL among them
        (1-based index) or return -1/0 to reject them all.
        
        We continue until only one candidate remains or none. Return final SQL or None.
        """

        # Extract the lists from the sample
        predict_sqls = []
        pred_results = []
        seen_results = []
        for predict_sql, pred_result in zip(sample['candidate_sqls'], sample['candidate_pred_results']):
            if 'Execution failed' not in str(pred_result) and 'too much time' not in str(pred_result) and not self.is_duplicated_execution_result(pred_result, seen_results):
                seen_results.append(pred_result)
                predict_sqls.append(re.sub(r'\s+', ' ', predict_sql).strip())
                pred_results.append(pred_result)

        while len(predict_sqls) > 1:
            new_predict_sqls = []
            new_pred_results = []

            # Split the current set of candidates into chunks of size 'max_candidates'
            for i in range(0, len(predict_sqls), max_candidates):
                chunk_sqls = predict_sqls[i : i + max_candidates]
                chunk_results = pred_results[i : i + max_candidates]

                if len(chunk_sqls) == 1:
                    new_predict_sqls.append(chunk_sqls[0])
                    new_pred_results.append(chunk_results[0])
                    continue

                # Build a temporary sample for just this chunk
                chunk_sample = deepcopy(sample)
                chunk_sample['candidate_sqls'] = chunk_sqls
                chunk_sample['candidate_pred_results'] = chunk_results

                # Ask the selection agent to pick the best SQL from 1..max_candidates
                # or return -1/0 to reject all
                prompt, answer_list = self.generate(chunk_sample)
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(prompt + answer_list[0])

                if not answer_list:
                    # If no answer from the agent, reject entire chunk
                    continue

                answer = self.extract_answer_index(answer_list[0])

                # If agent picks 1..len(chunk_sqls), keep that candidate
                if 1 <= answer <= len(chunk_sqls):
                    chosen_idx = answer - 1
                    new_predict_sqls.append(chunk_sqls[chosen_idx])
                    new_pred_results.append(chunk_results[chosen_idx])
                else:
                    # If agent picks -1 or 0 or out-of-range -> reject entire chunk
                    pass

            # Update lists with the "winners" of each chunk
            predict_sqls = new_predict_sqls
            pred_results = new_pred_results

            # If in this round we fail to keep anything, no final single solution
            if not predict_sqls:
                # get greedy answer = first predict_sqls
                # print('Greedy answer')
                return sample['candidate_sqls'][0]

        # At the end, if there's exactly one candidate, return it; otherwise None
        if len(predict_sqls) == 1:
            return predict_sqls[0]
    
        # print('greedy answer')
        return sample['candidate_sqls'][0]


class RankingAgent:
    def __init__(self, endpoint_type='llamacpp'):
        load_dotenv()

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm
        elif endpoint_type == 'openai':
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.get_answer = lambda x: get_answer_openai(client, x)

        self.prompt_template = """<|start_header_id|>user<|end_header_id|>
Given the question and following SQL queries, and execution results, please select the best SQL query that can answer the question. Answer the index of the SQL query you choose.

Question: {question}
Hint: {evidence}
"""
        self.choice_prompt = (
    "We have two candidate SQL queries:\n"
    "1) SQL:\n{sql1}\nResult:\n{res1}\n\n"
    "2) SQL:\n{sql2}\nResult:\n{res2}\n\n"
    "Which query is better for the question. Answer 1 or 2.\n"
)

    def build_prompt(self, sample):
        prompt = self.prompt_template.format(question=sample['question'], evidence=sample['evidence'])

        predict_sqls = sample['predict_sqls']
        pred_results = sample['pred_results']
        sql1 = predict_sqls[0]
        res1 = pred_results[0]
        if len(predict_sqls) > 1:
            sql2 = predict_sqls[1]
            res2 = pred_results[1]
        else:
            sql2 = ""
            res2 = ""

        choice_prompt = self.choice_prompt.format(
            sql1=sql1,
            res1=res1,
            sql2=sql2,
            res2=res2
        )

        # Combine with the base prompt
        prompt = prompt + choice_prompt
        prompt += """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""   
        return prompt

    def generate(self, sample):
        prompt = self.build_prompt(sample)
        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers



class ValidatorFixer():
    def __init__(self, endpoint_type='llamacpp'):
        self.prompt_template = """{schema}

Question: {question}
External knowledge: {evidence}

Generated SQL query: {sql_query}

Execution response:
{execution_response}

Feedback for the SQL query:
"""

    def parse_fixed_sql(self, answer: str):
        match = re.search(r'FIXED SQL: (.*)', answer, re.DOTALL)
        if match:
            fixed_sql = match.group(1).strip()
            return fixed_sql if fixed_sql.lower() != "none" else None
        return None
    
    def get_final_sql(self, predict_sql, fixed_sql, db_path):
        sqls_priority = [fixed_sql, predict_sql]
        sqls_priority = [sql for sql in sqls_priority if sql is not None]
        for sql in sqls_priority:
            # check if the sql is executable
            execution_error = check_sql_executability(sql, db_path)
            if execution_error is None:
                return sql
        return predict_sql

    def generate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])
            execution_result = _make_str_response(*execution_result)

        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sample['predict_sql'],
            execution_response=execution_result,
        )

        answers = self.get_answer([{"role": "user", "content": prompt}])
        # print(answers[0])
        fixed_sqls = [self.parse_fixed_sql(answer) for answer in answers]
        return prompt, answers, fixed_sqls, execution_result
    