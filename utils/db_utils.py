import os
import sqlite3

from pyserini.search.lucene import LuceneSearcher
import json
from func_timeout import func_set_timeout, FunctionTimedOut
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
import requests


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

# execute predicted sql with a time limitation
@func_set_timeout(30)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()

# execute predicted sql with a long time limitation (for buiding content index)
@func_set_timeout(2000)
def execute_sql_long_time_limitation(cursor, sql):
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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def detect_special_char(name):
    for special_char in ['(', '-', ')', ' ', '/']:
        if special_char in name:
            return True

    return False

def add_quotation_mark(s):
    return "`" + s + "`"

def get_column_contents(column_name, table_name, cursor):
    select_column_sql = "SELECT DISTINCT `{}` FROM `{}` WHERE `{}` IS NOT NULL LIMIT 2;".format(column_name, table_name, column_name)
    results = execute_sql_long_time_limitation(cursor, select_column_sql)
    column_contents = [str(result[0]).strip() for result in results]
    # remove empty and extremely-long contents
    column_contents = [content for content in column_contents if len(content) != 0 and len(content) <= 25]

    return column_contents

def get_db_schema_sequence(schema):
    schema_sequence = "database schema:\n"
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)
        
        # if table_comment != "":
        #     table_name += " ( comment : " + table_comment + " )"

        column_info_list = []
        for column_name, column_type, column_comment, column_content, pk_indicator, has_none_value in \
            zip(table["column_names"], table["column_types"], table["column_comments"], table["column_contents"], table["pk_indicators"], table["has_none_indicators"]):
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            additional_column_info = []
            # column type
            
            # pk indicator
            if pk_indicator != 0:
                additional_column_info.append("primary key")

            additional_column_info.append(f"type: {column_type}")
            # additional_column_info.append(f"{column_type}")
            # column comment
            if column_comment != "":
                additional_column_info.append("meaning: " + column_comment)
                # additional_column_info.append(column_comment)
            # representive column values
            if has_none_value:
                additional_column_info.append("has None")

            if len(column_content) != 0:
                additional_column_info.append("values: " + " , ".join(column_content))
            
            column_info_list.append(table_name + "." + column_name + " | " + " ; ".join(additional_column_info))
        
        schema_sequence += "table "+ table_name + " , columns = [\n  " + "\n  ".join(column_info_list) + "\n]\n"

    if len(schema["foreign_keys"]) != 0:
        schema_sequence += "foreign keys:\n"
        for foreign_key in schema["foreign_keys"]:
            for i in range(len(foreign_key)):
                if detect_special_char(foreign_key[i]):
                    foreign_key[i] = add_quotation_mark(foreign_key[i])
            schema_sequence += "{}.{} = {}.{}\n".format(foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3])
    else:
        schema_sequence += "foreign keys: None\n"
    return schema_sequence.strip()

def retrieve_most_similar_column_content(question, db_id, table_name, column_name):
    # requests to retrieval api to get most similar column content
    pass


def get_db_schema_sequence_with_matched_examples(schema, question):
    schema_sequence = "database schema:\n"
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)
        
        # if table_comment != "":
        #     table_name += " ( comment : " + table_comment + " )"

        column_info_list = []
        for column_name, column_type, column_comment, pk_indicator in \
            zip(table["column_names"], table["column_types"], table["column_comments"], table["pk_indicators"]):
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            additional_column_info = []
            # column type
            
            # pk indicator
            if pk_indicator != 0:
                additional_column_info.append("primary key")

            additional_column_info.append(f"type: {column_type}")
            # column comment
            if column_comment != "":
                additional_column_info.append("meaning: " + column_comment)
            # representive column values
            if len(column_content) != 0:
                additional_column_info.append("values: " + " , ".join(column_content))
            
            column_info_list.append(column_name + " | " + " ; ".join(additional_column_info))
        
        schema_sequence += "table "+ table_name + " , columns = [\n  " + "\n  ".join(column_info_list) + "\n]\n"

    if len(schema["foreign_keys"]) != 0:
        schema_sequence += "foreign keys:\n"
        for foreign_key in schema["foreign_keys"]:
            for i in range(len(foreign_key)):
                if detect_special_char(foreign_key[i]):
                    foreign_key[i] = add_quotation_mark(foreign_key[i])
            schema_sequence += "{}.{} = {}.{}\n".format(foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3])
    else:
        schema_sequence += "foreign keys: None\n"
    return schema_sequence.strip()

def get_matched_content_sequence(matched_contents):
    content_sequence = ""
    if len(matched_contents) != 0:
        content_sequence += "matched contents:\n"
        for tc_name, contents in matched_contents.items():
            table_name = tc_name.split(".")[0]
            column_name = tc_name.split(".")[1]
            if detect_special_char(table_name):
                table_name = add_quotation_mark(table_name)
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            
            content_sequence += table_name + "." + column_name + " ( " + " , ".join(contents) + " )\n"
    else:
        content_sequence = "matched contents: None"
    return content_sequence.strip()

def get_most_similar_column_contents(args):
    base_url, source, question, db_id, table_name, column_name = args
    # base_url = "http://localhost:8005"
    url = f"{base_url}/search_column_content"
    payload = {
        "source": source,
        "db_id": db_id,
        "table": table_name,
        "column": column_name,
        "query": question,
        "k": 2
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print("No results: ", source, db_id, table_name, column_name)
        return []

def get_db_schema(api_url, source, question, db_path, db_comments, db_id):
    if db_id in db_comments:
        db_comment = db_comments[db_id]
    else:
        db_comment = None

    cursor = get_cursor_from_path(db_path)
    
    # obtain table names
    results = execute_sql(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [result[0].lower() for result in results]

    schema = dict()
    schema["schema_items"] = []
    foreign_keys = []
    # for each table
    for table_name in table_names:
        # skip SQLite system table: sqlite_sequence
        if table_name == "sqlite_sequence":
            continue
        # obtain column names in the current table
        results = execute_sql(cursor, "SELECT name, type, pk FROM PRAGMA_TABLE_INFO('{}')".format(table_name))
        column_names_in_one_table = [result[0].lower() for result in results]
        column_types_in_one_table = [result[1].lower() for result in results]
        pk_indicators_in_one_table = [result[2] for result in results]

        with ThreadPool(processes=16) as pool:
            column_contents = pool.map(get_most_similar_column_contents, [(api_url, source, question, db_id, table_name, column_name) for column_name in column_names_in_one_table])

        # obtain foreign keys in the current table
        results = execute_sql(cursor, "SELECT * FROM pragma_foreign_key_list('{}');".format(table_name))
        for result in results:
            if None not in [result[3], result[2], result[4]]:
                foreign_keys.append([table_name.lower(), result[3].lower(), result[2].lower(), result[4].lower()])
        
        # obtain comments for each schema item
        if db_comment is not None:
            if table_name in db_comment: # record comments for tables and columns
                table_comment = db_comment[table_name]["table_comment"]
                column_comments = [db_comment[table_name]["column_comments"][column_name] \
                    if column_name in db_comment[table_name]["column_comments"] else "" \
                        for column_name in column_names_in_one_table]
            else: # current database has comment information, but the current table does not
                table_comment = ""
                column_comments = ["" for _ in column_names_in_one_table]
        else: # current database has no comment information
            table_comment = ""
            column_comments = ["" for _ in column_names_in_one_table]

        has_none_indicators = []
        for column_name in column_names_in_one_table:
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}` WHERE `{column_name}` IS NULL")
            count = cursor.fetchone()[0]
            if count > 0:
                has_none_indicators.append(1)
            else:
                has_none_indicators.append(0)

        schema["schema_items"].append({
            "table_name": table_name,
            "table_comment": table_comment,
            "column_names": column_names_in_one_table,
            "column_types": column_types_in_one_table,
            "column_comments": column_comments,
            "column_contents": column_contents,
            "pk_indicators": pk_indicators_in_one_table,
            "has_none_indicators": has_none_indicators
        })
    
    schema["foreign_keys"] = foreign_keys
    
    return schema
