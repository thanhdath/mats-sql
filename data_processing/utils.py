from sql_metadata import Parser
import re
import os
import sqlparse

def remove_table_alias(s):
    try:
        tables_aliases = Parser(s).tables_aliases
    except Exception as e:
        return s

    new_tables_aliases = {}
    for i in range(1,11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
    
    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        # remove AS clauses
        s = s.replace("AS " + k + " ", "")
        # replace table alias with thier original names
        s = s.replace(k, v)
    
    return s

def extract_select_clause(sql_query):
    # Define a regex pattern to match the SELECT clause up to the FROM keyword
    pattern = re.compile(r"SELECT\s.*?\s(?=FROM)", re.IGNORECASE | re.DOTALL)
    
    # Search for the pattern in the SQL query
    match = pattern.search(sql_query)
    
    if match:
        # Return the matched portion (SELECT clause)
        return match.group(0).strip()
    else:
        # Return None if no match is found
        return None
    
def get_table_columns_list(schema):
    columns = []
    for table_data in schema['schema_items']:
        table_name = table_data['table_name'].lower()
        column_names = table_data['column_names']
        column_names = [x.lower() for x in column_names]

        for column in column_names:
            columns.append(f"{table_name}.{column}")
            columns.append(f"{table_name}.`{column}`")
            columns.append(column)

    columns = list(set(columns))
    return columns

def get_columns_in_select_clause(sql_query, schema):
    column_list = get_table_columns_list(schema)
    select_clause = extract_select_clause(sql_query)

    select_clause = remove_table_alias(sqlparse.format(select_clause, keyword_case = "upper", identifier_case = "lower"))

    try:
        sql_tokens = [token.value for token in Parser(select_clause.lower()).tokens]
    except Exception as e:
        sql_tokens = sql_query.lower().split()

    select_columns = []
    for token in sql_tokens:
        if token in column_list:
            select_columns.append(token)
    return select_columns


def get_equation_function_in_select_clause(sql_query):
    """
    equation function includes min, max, avg, sum, count, divide, +, /, case when
    """
    select_clause = extract_select_clause(sql_query)
    norm_select_clause = remove_table_alias(sqlparse.format(select_clause, keyword_case = "upper", identifier_case = "lower"))

    try:
        sql_tokens = [token.value for token in Parser(norm_select_clause.lower()).tokens]
    except Exception as e:
        sql_tokens = norm_select_clause.lower().split()

    equation_functions = []
    for token in sql_tokens:
        if token in ["min", "max", "avg", "sum", "count", "divide", "+", "/", "case", "when"]:
            equation_functions.append(token)

    return equation_functions

def remove_tables_from_columns(columns):
    new_columns = []
    for col in columns:
        new_columns.append(col.split('.')[-1])
    return new_columns

def check_columns_match(true_columns, pred_columns):
    true_columns = remove_tables_from_columns(true_columns)
    pred_columns = remove_tables_from_columns(pred_columns)

    # classify error types, unnecessary columns, missing columns, wrong order, return a string of error type
    if true_columns == pred_columns:
        return 'correct'
    else:
        if set(true_columns) == set(pred_columns):
            return 'incorrect: wrong order'
        elif set(true_columns) - set(pred_columns):
            return 'incorrect: missing columns ' + str(set(true_columns) - set(pred_columns))
        elif set(pred_columns) - set(true_columns):
            return 'incorrect: unnecessary columns ' + str(set(pred_columns) - set(true_columns))
        


from sql_metadata import Parser
import re
def normalization(sql):
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(
            r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)

        return s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    return processing_func(sql)

def norm_sql_query(true_sql, schema):
    try:
        norm_sql = normalization(true_sql).strip()
    except Exception as err: 
        print("Error sql: ", true_sql)
        print(err)
        return true_sql
    
    sql_tokens = norm_sql.split()

    for table in schema['schema_items']:
        if table['table_name'] in sql_tokens:  # for used tables
            columns = table['column_names']
            table_name = table['table_name']

            if " " in table_name:
                true_sql = re.sub(rf"(?<=[ \.]){table_name}(?=[ ,\.])", f"`{table_name}`", true_sql)

            for column_name in columns:
                if column_name.lower() in sql_tokens or table_name.lower() + "." + column_name.lower() in sql_tokens:
                    # use regex, if column_name_original is wrapped by spaces then replace it with `column_name_original`
                    if column_name in true_sql:
                        true_sql = re.sub(rf"(?<=[ \.]){column_name}(?=[ ,])", f"`{column_name}`", true_sql)
                    elif column_name.lower() in true_sql:
                        true_sql = re.sub(rf"(?<=[ \.]){column_name.lower()}(?=[ ,])", f"`{column_name}`", true_sql)
    return true_sql
