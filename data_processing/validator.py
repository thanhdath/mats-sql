
import sqlite3
import multiprocessing.pool
import functools
import pandas as pd
import re
import sqlparse
from sql_metadata import Parser
from utils import get_table_columns_list, remove_table_alias, get_columns_in_select_clause, get_equation_function_in_select_clause, remove_table_alias

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
    if type(true_response) == str and type(pred_response) == str:
        return true_response == pred_response
    elif type(true_response) == str and type(pred_response) != str:
        return False
    elif type(true_response) != str and type(pred_response) == str:
        return False
    else:
        return set([tuple(x) for x in true_response.values.tolist()]) == set([tuple(x) for x in pred_response.values.tolist()])

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
#         use_beam_search=True,
#         n=4,
#         stop=['=========']
#         # eos_token_id=self.tokenizer.convert_tokens_to_ids(['<|end|>'])
#     )
#     response = response.choices[0].text
#     return response
    
def get_answer_vllm(messages):
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


def get_answer_llamacpp(messages):
    import requests
    response = requests.post("http://localhost:8000/v1/completions",
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct/",
                "prompt": messages[0]['content'],
                "n_predict": 256,
                "stop": ["========="]
                }).json()
    return response["content"]
    
class ValidatorSelect:
    def __init__(self, endpoint_type='llamacpp'):
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.max_columns', 10)

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm

        self.prompt_template = open('./few_shot_prompt_select.txt').read() + """=========
{schema}

Matched contents are written in this format table.column (some values can be found in that column)
{matched_content}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Feedback:
SELECT.
1. Based on the SQL query, the query selects: {select_columns}"""



    def check_able_to_comment(self, sql_query):
        equations = get_equation_function_in_select_clause(sql_query)
        if len(equations) == 0:
            return True

        able_to_comment_equations = ['min', 'max', 'sum', 'avg', 'divide', '+', '/']
        # if equation doesn't contain any other than the above, then can comment
        for equation in equations:
            if equation not in able_to_comment_equations:
                return False
        
        return True

    def comment(self, sql, sample, execution_result):
        try:
            select_columns = get_columns_in_select_clause(sql, sample['schema'])
            if len(select_columns) == 0:
                select_columns = ""
        except:
            select_columns = ""

        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            matched_content=sample['content_sequence'],
            question=sample['text'],
            sql_query=sql,
            execution_response=_make_str_response(*execution_result),
            select_columns=select_columns
        )
        answer = prompt.split("Feedback:")[-1] + self.get_answer([{"role": "user", "content": prompt}])
        return answer

    def validate(self, sample):
        able_to_comment = self.check_able_to_comment(sample['predict_sql'])
        execution_result = _execute_sql("../" + sample['db_path'], sample['predict_sql'])
        if able_to_comment:
            # generate comment using few-shot prompting
            answer = self.comment(sample['predict_sql'], sample, execution_result)
            return answer, execution_result
        else:
            return None, execution_result
        

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
    
class FixAgent:
    def __init__(self, prompt_template, endpoint_type='llamacpp'):
        self.prompt_template = prompt_template

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            self.get_answer = get_answer_vllm

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
        c.execute(query)
        result = c.fetchall()
        conn.close()

        if result[0][0] > 0:
            return True
        else:
            return False
        
    def validate(self, sample):
        execution_result = _execute_sql("../" + sample['db_path'], sample['predict_sql'])

        order_columns, order_by_clause = self.get_columns_in_order_clause(sample['predict_sql'], sample['schema'])
        if order_columns is not None and len(order_columns) > 0:
            column = order_columns[0]
            if self.check_order_by_column_has_none_values(column, "../" + sample['db_path']):
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
            else:
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
    