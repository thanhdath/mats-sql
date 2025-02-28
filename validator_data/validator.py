
import sqlite3
import multiprocessing.pool
import functools
import re
import sqlparse
import requests
from sql_metadata import Parser
from validator_data.utils import get_table_columns_list, remove_table_alias, get_columns_in_select_clause, get_equation_function_in_select_clause, remove_table_alias
from openai import OpenAI
import os
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
import time

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 10)

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

def _execute_sql_with_timeout(db_path, action):
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    actions = action.split(";")
    actions = [x for x in actions if len(x.strip()) > 0]
    
    if len(actions) == 0:
        return "No SQL query executed.", True
    
    cursor = conn.cursor()
    for action in actions:
        try:
            # Use pandas to execute the query and fetch the result
            response = pd.read_sql_query(action, conn)
            has_error = False
        except Exception as error:
            # If the SQL query is invalid, return the error message from sqlite
            response = str(error)
            has_error = True
            cursor.close()
            break
    
    cursor.close()
    conn.close()
    return response, has_error

def _execute_sql(db_path, sql_query, timeout=15):
    try:
        # Use func_timeout to enforce the timeout
        pred_result, has_error = func_timeout(timeout, _execute_sql_with_timeout, args=(db_path, sql_query))
    except FunctionTimedOut:
        pred_result = "The query takes too much time."
        has_error = True
    except Exception as err:
        pred_result = str(err)
        has_error = True
    return pred_result, has_error

def execute_sql_with_time(db_path, sql_query, timeout=10):
    start_time = time.time()
    try:
        # Use func_timeout to enforce the timeout
        pred_result, has_error = func_timeout(timeout, _execute_sql_with_timeout, args=(db_path, sql_query))
    except FunctionTimedOut:
        pred_result = "The query takes too much time."
        has_error = True
    except Exception as err:
        pred_result = str(err)
        has_error = True
    execution_time = time.time() - start_time
    return pred_result, has_error, execution_time

def _make_str_response(response, has_error, add_num_duplicated=False):
    if has_error:
        response = str(response)
        elms = response.split(":")
        response = ":".join(elms[-2:])
        return response
    else:     
        # df = pd.DataFrame(response)
        # return str(df)
        str_response = str(response).strip()
        if add_num_duplicated:
            num_duplicated = response.duplicated().sum()
            str_response += f"\nNumber of duplicated records: {num_duplicated}."

        return str_response
    
def is_execution_correct(true_response, pred_response):
    if type(true_response) == str and type(pred_response) == str:
        return true_response == pred_response
    elif type(true_response) == str and type(pred_response) != str:
        return False
    elif type(true_response) != str and type(pred_response) == str:
        return False
    else:
        return set([tuple(x) for x in true_response.values.tolist()]) == set([tuple(x) for x in pred_response.values.tolist()])

def get_answer_vllm(messages):
    response = requests.post("http://localhost:8003/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-14B-Instruct/",
                "prompt": messages[0]['content'],
                "max_tokens": 1024,
                "use_beam_search": True,
                "n": 4,
                "temperature": 0.0,
                "stop": ["========"]
                }).json()
    # print(response)
    return response["choices"][0]["text"]


def get_answer_llamacpp(messages):
    response = requests.post("http://localhost:8000/v1/completions",
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct/",
                "prompt": messages[0]['content'],
                "n_predict": 256,
                "stop": ["========="]
                }).json()
    return response["content"]

class Validator:
    def __init__(self, endpoint_type='llamacpp'):
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.max_columns', 10)

        if endpoint_type == 'llamacpp':
            self.get_answer = get_answer_llamacpp
        elif endpoint_type == 'vllm':
            # self.get_answer = get_answer_vllm
            client = OpenAI(
                base_url="http://localhost:8005/v1",
                api_key="no-key",
            )
            self.get_answer = lambda x: get_answer_openai(client, x, model='fixed')

        elif endpoint_type == 'openai':
            from dotenv import load_dotenv
            load_dotenv()
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.get_answer = lambda x: get_answer_openai(client, x)

    def process_feedback_message_from_completion(self, prompt, answer):
        if prompt is None:
            prompt = ''
        
        if answer is None:
            return f"{self.first_token}\nNone"

        answer = prompt.split("Feedback:")[-1] + answer
        answer = answer.replace('<|assistant|>', '').replace('<|end|>', '').strip()
        answer = answer.replace('<|start_header_id|>assistant<|end_header_id|>', '').replace('<|eot_id|>', '').strip()
        return answer
    
class ValidatorSelect(Validator):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(endpoint_type=endpoint_type)
        self.first_token = "SELECT."

        self.prompt_template = open('./validator_data/few_shot_prompt_select.txt').read() + """=========
{schema}

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

        able_to_comment_equations = ['min', 'max', 'sum', 'avg', 'divide', '+', '/', 'count']
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
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sql,
            execution_response=_make_str_response(execution_result[0], execution_result[1], add_num_duplicated=True),
            select_columns=select_columns
        )

        # answers = [
        #     prompt.split("Feedback:")[-1] + answer for answer in self.get_answer([{"role": "user", "content": prompt}])
        # ]
        answers = self.get_answer([{"role": "user", "content": prompt}])

        return prompt, answers

    def validate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])

        able_to_comment = self.check_able_to_comment(sample['predict_sql'])
        if able_to_comment:
            # generate comment using few-shot prompting
            prompt, answers = self.comment(sample['predict_sql'], sample, execution_result)
            return prompt, answers, execution_result
        else:
            return None, [None], execution_result
        

class ValidatorJOIN(Validator):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(endpoint_type=endpoint_type)
        self.first_token = "JOIN."

        self.prompt_template = open('./validator_data/few_shot_prompt_join.txt').read() + """
=========
{schema}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Strictly follow examples format.
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
                return ''

    def get_used_fks(self, sql_query):
        # use re, get all condition join after ON
        pattern = re.compile(r" ON\s.*?(?=\sWHERE|\sORDER BY|\sLIMIT|\sGROUP BY)", re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(sql_query)
        all_used_fks = []
    
        # Pattern to extract the entire 'src_table.src_col = trg_table.trg_col' as a single string, handle this case also frpm.`school code` = schools.cdscode
        # fk_pattern = re.compile(r'(\w+\.\w+\s*=\s*\w+\.\w+)', re.IGNORECASE)
        fk_pattern = re.compile(
            r'([`"]?[a-zA-Z0-9_]+[`"]?\.[`"]?[a-zA-Z0-9_ ]+[`"]?\s*=\s*[`"]?[a-zA-Z0-9_]+[`"]?\.[`"]?[a-zA-Z0-9_ ]+[`"]?)',
            re.IGNORECASE
        )

        for match in matches:
            # Extract all foreign key conditions from the matched ON clause
            fks = fk_pattern.findall(match)
            if fks:
                all_used_fks.extend(fks)
        
        return all_used_fks

 

    def get_tables_in_join_clause(self, sql_query, schema):
        table_list = self.get_table_list(schema)
        sql_query = remove_table_alias(sqlparse.format(sql_query, keyword_case = "upper", identifier_case = "lower"))
        join_clause = self.extract_join_clause(sql_query)

        used_tables = []
        for token in join_clause.strip(';').split():
            if token in table_list:
                used_tables.append(token)

        used_fks = self.get_used_fks(sql_query)
        return used_tables, used_fks
    
    def add_prompt_used_fk_not_exist(self, used_tables, used_fks, sample):
        foreign_keys = sample['schema']['foreign_keys']
        exist_fks = {}
        for src_table, src_col, trg_table, trg_col in foreign_keys:
            # exist_fks.append((src_table, src_col, trg_table, trg_col))
            # exist_fks.append((trg_table, trg_col, src_table, src_col))
            if (src_table, trg_table) not in exist_fks:
                exist_fks[(src_table, trg_table)] = []
                exist_fks[(trg_table, src_table)] = []
            exist_fks[(src_table, trg_table)].append((src_col, trg_col))
            exist_fks[(trg_table, src_table)].append((trg_col, src_col))
        
        added_prompt = ""
        used_tables_in_fks = set()
        for fk in used_fks:
            src, trg = fk.split("=")
            src_table, src_col = src.strip().split(".")
            trg_table, trg_col = trg.strip().split(".")
            used_tables_in_fks.add(src_table)
            used_tables_in_fks.add(trg_table)
            # if (src_table, src_col, trg_table, trg_col) not in exist_fks:
            if (src_table, trg_table) not in exist_fks:
                added_prompt += f"\n- The foreign key `{src_table}.{src_col} = {trg_table}.{trg_col}` does not exist in the schema, the query is incorrect. Need to add more tables to the query."
            elif (src_col, trg_col) not in exist_fks[(src_table, trg_table)]:
                correct_fk = exist_fks[(src_table, trg_table)][0]
                added_prompt += f"\n- The foreign key `{src_table}.{src_col} = {trg_table}.{trg_col}` does not exist in the schema, the query is incorrect. The query need to use foreign key `{src_table}.{correct_fk[0]} = {trg_table}.{correct_fk[1]}"
        
        # 
        unincluded_tables = set(used_tables_in_fks) - set(used_tables)
        if len(unincluded_tables) > 0:
            added_prompt += f"\n - The query is incorrect. Please add the tables {list(unincluded_tables)} to the FROM statement."
        
        return added_prompt


    def validate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])
        used_tables, used_fks = self.get_tables_in_join_clause(sample['predict_sql'], sample['schema'])
        # parse sche
        added_prompt = self.add_prompt_used_fk_not_exist(used_tables, used_fks, sample)

        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sample['predict_sql'],
            execution_response=_make_str_response(*execution_result),
            used_tables=used_tables,
            used_fks=used_fks
        ).strip() + added_prompt + "\n- Based on the question, the query should use tables"

        # answers = [
        #     prompt.split("Feedback:")[-1] + answer for answer in self.get_answer([{"role": "user", "content": prompt}])
        # ]
        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers, execution_result

class ValidatorOrder(Validator):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(endpoint_type=endpoint_type)
        self.first_token = "ORDER BY."

        self.prompt_no_none = open('./validator_data/few_shot_prompt_order.txt').read().replace("{", "{{").replace("}", "}}") + """
=========
{schema}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Feedback:
ORDER BY.
- The SQL query uses ```{order_by_clause}```.
- Based on the question, the query should use"""

        self.prompt_has_none = open('./validator_data/few_shot_prompt_order.txt').read().replace("{", "{{").replace("}", "}}") + """
=========
{schema}

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
            order_by_clause = order_by_clause.group(1)
            order_by_clause = re.sub("\s+", " ", order_by_clause)
            return order_by_clause

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

        # print('Order by clause:', order_by_clause)

        if len(equation_functions) > 0:
            # print('Equation functions:', equation_functions)
            return None, order_by_clause # not supported yet
        else:
            columns = []
            # print('Order clause tokens:', order_clause_tokens)
            # print('column list:', column_list)
            for token in order_clause_tokens:
                if token in column_list:
                    columns.append(token)

            # norm columns list, add table.column if '.' not present. table can extract using regex on sql query SELECT x FROM table
            norm_columns = []
            for column in columns:
                if "." not in column:
                    # regex find table name right after the word 'FROM', table name can be wrapped inside ``
                    try:
                        table = re.search(r'(?i)FROM\s+`?(\w+)`?', sql_query).group(1)
                        norm_columns.append(f"{table}.{column}")
                    except:
                        norm_columns.append(column)
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
        elms = column.split(".")
        if len(elms) == 1:
            return False
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
        
    def validate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])

        order_columns, order_by_clause = self.get_columns_in_order_clause(sample['predict_sql'], sample['schema'])
        if order_columns is not None and len(order_columns) > 0:
            column = order_columns[0]

            if self.check_order_by_column_has_none_values(column, "./" + sample['db_path']) == True:
                prompt = self.prompt_has_none.format(
                    schema=sample['schema_sequence'], 
                    question=sample['question'],
                    evidence=sample['evidence'],
                    sql_query=sample['predict_sql'],
                    execution_response=_make_str_response(*execution_result),
                    order_by_clause=order_by_clause,
                    order_by_column=column
                )
                # answers = [prompt.split("Feedback:")[-1]]
                answers = []
                return None, answers, execution_result
            else: # False or error string
                # print(column)
                table, column = column.split(".")
                # if "desc limit 1" in order_by_clause.lower():
                #     new_order_clause = f"Please replace Order by with this clause in the query `{table}`.`{column}` = (SELECT MAX(`{table}`.`{column}`) FROM `{table}`).\nConclude: incorrect."
                #     prompt = None
                #     answers = [new_order_clause]
                # elif "limit 1" in order_by_clause.lower():
                #     new_order_clause = f"Please replace Order by with this clause in the query `{table}`.`{column}` = (SELECT MIN(`{table}`.`{column}`) FROM `{table}`);\nConclude: incorrect."
                #     answers = [new_order_clause]
                #     prompt = None
                # else:
                if True:
                    prompt = self.prompt_no_none.format(
                        schema=sample['schema_sequence'], 
                        question=sample['question'],
                        evidence=sample['evidence'],
                        sql_query=sample['predict_sql'],
                        execution_response=_make_str_response(*execution_result),
                        order_by_clause=order_by_clause)
                    # answers = [
                    #     prompt.split("Feedback:")[-1] + answer for answer in self.get_answer([{"role": "user", "content": prompt}])
                    # ]
                    answers = self.get_answer([{"role": "user", "content": prompt}])
                    
        else:
            answers = []
            prompt = None
            
        return prompt, answers, execution_result

def get_answer_openai(client, messages, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    response = response.choices[0].message.content.strip()
    return [response]

    

class ValidatorCondition(Validator):
    def __init__(self, prompt_file='./validator_data/few_shot_prompt_condition.txt', endpoint_type='llamacpp'):
        super().__init__(endpoint_type=endpoint_type)
        self.first_token = "CONDITION."

        self.prompt_template = open(prompt_file).read() + """
=========
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response [written in pandas format].
{execution_response}

If the execution response empty response, it is incorrect. Add your thought to the end of the feedback to modify the query.
If there is a syntax error, write "Conclude: incorrect", then write the reason and guide to fix it.
Some error and how to fix:
- no such column, guide to add need tables in the JOIN.
- no such table, need write a correct table name.
Always add "Conclude: correct." or "Conclude: incorrect." at the end of the feedback.

Feedback:
CONDITION.
"""
       
    def get_table_list(self, schema):
        tables = []
        for table_data in schema['schema_items']:
            table_name = table_data['table_name'].lower()
            tables.append(table_name)
        tables = list(set(tables))
        return tables
    
    def extract_condition_clause(self, sql_query):
        # extract conditions after WHERE and before group by, having, order by
        pattern = re.compile(r"WHERE\s.*?(?=\sGROUP BY|\sHAVING|\sORDER BY|\sLIMIT)", re.IGNORECASE | re.DOTALL)
        match = pattern.search(sql_query)
        if match:
            return match.group(0).strip()
        else:
            # found None, extract conditions to the end of the sql query
            pattern = re.compile(r"WHERE\s.*", re.IGNORECASE | re.DOTALL)
            match = pattern.search(sql_query)
            if match:
                return match.group(0).strip()
            else:
                return None
            
    def has_column_with_more_than_20_percent_none(self, execution_result):
        import pandas as pd  # Ensure pandas is imported
        
        # Check if execution_result is a string or None (indicating an error or empty response)
        if isinstance(execution_result, str) or execution_result is None:
            return True
        # Check if execution_result is a DataFrame
        elif isinstance(execution_result, pd.DataFrame):
            # Check if the DataFrame is empty
            if execution_result.empty:
                return True
            # Check if the DataFrame has only one element with value 0
            if execution_result.size == 1 and execution_result.values[0][0] == 0:
                return True
            # Calculate the fraction of None (NaN) values in each column
            missing_ratios = execution_result.isnull().mean()
            # Check if any column has more than 20% None values
            return any(missing_ratios >= 0.2)
        else:
            # If execution_result is not a DataFrame or string, consider it invalid
            return True
    

    def validate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])


        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sample['predict_sql'],
            execution_response=_make_str_response(*execution_result),
        )
  
        answers = self.get_answer([{"role": "user", "content": prompt}])

        return prompt, answers, execution_result


class ValidatorConditionWithTrueSQL(ValidatorCondition):
    def __init__(self, prompt_file='./validator_data/few_shot_prompt_condition.txt', endpoint_type='llamacpp'):
        super().__init__(endpoint_type=endpoint_type)
        self.first_token = "CONDITION."

        self.prompt_template = open(prompt_file).read() + """
=========
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response [written in pandas format].
{execution_response}

If the execution response empty response, it is incorrect. Add your thought to the end of the feedback to modify the query.
If there is a syntax error, write "Conclude: incorrect", then write the reason and guide to fix it.
Some error and how to fix:
- no such column, guide to add need tables in the JOIN.
- no such table, need write a correct table name.
Always add "Conclude: correct." or "Conclude: incorrect." at the end of the feedback.

Use this hidden True SQL query to write correct analysis that derives to the correct answer. The True SQL query cannot be used in the analysis.
Hidden True SQL query: {true_sql_query} 

Feedback:
CONDITION.
"""

    def validate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])

        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sample['predict_sql'],
            execution_response=_make_str_response(*execution_result),
            true_sql_query=sample['sql'],
        )
  
        answers = self.get_answer([{"role": "user", "content": prompt}])

        return prompt, answers, execution_result
    

class ValidatorJOINWithTrueSQL(ValidatorJOIN):
    def __init__(self, endpoint_type='llamacpp'):
        super().__init__(endpoint_type=endpoint_type)
        self.first_token = "JOIN."

        self.prompt_template = open('./validator_data/few_shot_prompt_join.txt').read() + """
=========
{schema}

Question: {question}

SQL query: {sql_query}

Execution response [written in pandas format]:
{execution_response}

Use this hidden True SQL query to write correct analysis that derives to the correct answer. The True SQL query cannot be used in the analysis.
Hidden True SQL query: {true_sql_query} 

Strictly follow examples format.
Feedback:
JOIN.
- The SQL query uses tables {used_tables}, joining them on foreign keys {used_fks}."""

    def validate(self, sample, execution_result=None):
        if execution_result is None:
            execution_result = _execute_sql("./" + sample['db_path'], sample['predict_sql'])
        used_tables, used_fks = self.get_tables_in_join_clause(sample['predict_sql'], sample['schema'])
        # parse sche
        added_prompt = self.add_prompt_used_fk_not_exist(used_tables, used_fks, sample)

        prompt = self.prompt_template.format(
            schema=sample['schema_sequence'], 
            question=sample['question'],
            evidence=sample['evidence'],
            sql_query=sample['predict_sql'],
            execution_response=_make_str_response(*execution_result),
            true_sql_query=sample['sql'],
            used_tables=used_tables,
            used_fks=used_fks
        ).strip() + added_prompt + "\n- Based on the question, the query should use tables"

        answers = self.get_answer([{"role": "user", "content": prompt}])
        return prompt, answers, execution_result
    

