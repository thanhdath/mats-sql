import os
import argparse
import json
import re
from utils.db_utils import check_sql_executability

def extract_sql_in_code_block(pred_sql_text):
    """
    Extracts the SQL query from a text block that contains code block marked by triple backticks (```sql ... ```).
    
    Args:
        pred_sql_text (str): The input text that may contain a SQL code block.
    
    Returns:
        str: The extracted SQL query or an empty string if no SQL code block is found.
    """
    # Use regex to search for the SQL code block enclosed in triple backticks
    sql_block_match = re.search(r"```sql\s+(.+?)\s+```", pred_sql_text, re.DOTALL)

    if sql_block_match:
        # Extract the SQL query from the matched block
        sql_query = sql_block_match.group(1).strip()
        return sql_query
    else:
        return pred_sql_text

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, default='data/evaluate/phi-3-planner_spider_dev.jsonl')
parser.add_argument('--planner_only', action='store_true')
args = parser.parse_args()

if "spider_dev" in args.pred_file:
    args.dev_file = "data/sft_data_collections/spider/dev.json"
elif "spider_dk" in args.pred_file:
    args.dev_file = 'data/sft_spider_dk_text2sql.json'
elif "spider_realistic" in args.pred_file:
    args.dev_file = 'data/sft_spider_realistic_text2sql.json'
elif "spider_syn" in args.pred_file:
    args.dev_file = 'data/sft_spider_syn_text2sql.json'
elif "dr_spider" in args.pred_file:
    args.dev_file = 'data/sft_dr_spider_text2sql.json'
elif "bird" in args.pred_file and 'dev' in args.pred_file:
    args.dev_file = "data/full_value_matching_schema_insight_bird_062024_with_evidence_dev_text2sql.json"
elif "bird_train" in args.pred_file:
    args.dev_file = "data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json"

# Load dev_file to get the correct order of (db_id, question) keys
with open(args.dev_file) as dev_fp:
    dev_data = json.load(dev_fp)

# Create a list of keys from dev_file to sort by
dev_keys = [f"{sample['db_id']} {sample['question']}" for sample in dev_data]

def get_final_sql(predict_sql, fixed_sql, db_path):
    sqls_priority = [fixed_sql, predict_sql]
    sqls_priority = [sql for sql in sqls_priority if sql is not None]
    for sql in sqls_priority:
        # check if the sql is executable
        execution_error = check_sql_executability(sql, db_path)
        if execution_error is None:
            return sql
    return predict_sql

def get_predict_sqls():
    # Load predictions and store them using the same (db_id, question) key
    predicted_sqls_dict = {}
    with open(args.pred_file) as pred_fp:
        for line in pred_fp:
            sample = json.loads(line)
            db_id = sample.get('db_id')
            question = sample.get('question')

            key = f"{db_id} {question}"

            predict_sqls = []
            if not args.planner_only:
                predict_sqls = [x for x in sample.get('fixed_sqls', []) if x is not None]
                predict_sqls = [extract_sql_in_code_block(x) for x in predict_sqls]
            predict_sqls.extend(sample['predict_sqls'])

            if 'final_sql' in sample:
                predict_sqls = [sample['final_sql']]
            elif 'fixed_sqls' in sample:
                pair_sqls = [(x, y) for x, y in zip(sample['predict_sqls'], sample['fixed_sqls'])]
                predict_sqls = [get_final_sql(x, y, sample['db_path']) for x, y in pair_sqls]


            predict_sqls = [re.sub(r'\border\b(?!\s+BY)', r'`order`', query).replace("``", "`") for query in predict_sqls]

            pred_sql = None
            for sql in predict_sqls:
                execution_error = check_sql_executability(sql, sample["db_path"])
                if execution_error is not None:
                    if "syntax error" in execution_error:
                        print(sql)
                    continue
                
                pred_sql = sql
                break
            if pred_sql is None:
                pred_sql = predict_sqls[0]

            pred_sql = pred_sql.replace("\n", " ").strip()
            if len(pred_sql.strip()) == 0:
                pred_sql = "SELECT;"
 
            # Store the prediction in the dictionary using the key
            predicted_sqls_dict[key] = (pred_sql, sample['db_id'])

    return predicted_sqls_dict


if 'spider' in args.pred_file:
    predicted_sqls_dict = get_predict_sqls()
    predicted_sqls = [predicted_sqls_dict.get(key, ("SELECT", ""))[0] for key in dev_keys]
elif 'bird' in args.pred_file:
    predicted_sqls_dict = get_predict_sqls()
    bird_results_dict = {}
    for idx, key in enumerate(dev_keys):
        pred_sql, db_id = predicted_sqls_dict.get(key, ("SELECT", ""))
        bird_results_dict[idx] = pred_sql + "\t----- bird -----\t" + db_id

if "bird" in args.pred_file:
    with open("predict_dev.json", "w", encoding = 'utf-8') as f:
        f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))
    os.system("sh bird_evaluation/run_evaluation.sh predict_dev.json")
elif "spider_dev" in args.pred_file:
    with open("pred_sqls.txt", "w+", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')

    with open("pred_sqls.txt", "w+", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Test suit execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')
elif "spider_dk" in args.pred_file:
    with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/Spider-DK/dk_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
elif "spider_realistic" in args.pred_file:
    with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider-realistic/realistic_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
    with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Test suit execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider-realistic/realistic_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')
elif "spider_syn" in args.pred_file:
    with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/Spider-Syn/Spider-Syn/syn_dev_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
    with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print("Test suit execution accuracy:")
    os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/Spider-Syn/Spider-Syn/syn_dev_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')
elif "dr_spider" in args.pred_file:
    test_set_names = [
        # DB Schema related
        'DB_schema_synonym',
        'DB_schema_abbreviation',
        'DB_DBcontent_equivalence',

        # NLQ related
        'NLQ_keyword_synonym',
        'NLQ_keyword_carrier',
        'NLQ_column_synonym',
        'NLQ_column_carrier',
        'NLQ_column_attribute',
        'NLQ_column_value',
        'NLQ_value_synonym',
        'NLQ_multitype',
        'NLQ_others',
        # SQL related
        'SQL_comparison',
        'SQL_sort_order',
        'SQL_NonDB_number',
        'SQL_DB_text',
        'SQL_DB_number',
    ]

    #test_set_names = ['DB_schema_synonym', 'DB_schema_abbreviation', 'DB_DBcontent_equivalence']
    raw_dataset = json.load(open(args.dev_file))
    for test_set_name in test_set_names:
        
        if test_set_name not in args.pred_file:
            continue
        print(test_set_name)
        test_set_predicted_sqls = [predicted_sql for predicted_sql, raw_data in zip(predicted_sqls, raw_dataset) if raw_data["source"] == "dr.spider-" + test_set_name]

        database_file_path = "database_post_perturbation" if test_set_name.startswith("DB_") else "databases"
        db_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, database_file_path)
        gold_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "gold_post_perturbation.sql")

        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in test_set_predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold {} --pred pred_sqls.txt --db {} --etype exec'.format(gold_path, db_path))
        
