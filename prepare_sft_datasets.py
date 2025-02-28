import json
import os
import re
import random
import sqlparse
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk import ngrams
from sql_metadata import Parser
from utils.db_utils import get_db_schema
import subprocess

random.seed(42)

def extract_large_numbers(text):
    number_information = []
    patterns = {
        'thousand': 10**3,
        'million': 10**6,
        'billion': 10**9,
        'trillion': 10**12
    }
    
    for word, multiplier in patterns.items():
        matches = re.findall(r'(\d+\.?\d*)\s*{}'.format(word), text, flags=re.IGNORECASE)
        for match in matches:
            number = float(match) * multiplier
            number_information.append(match + " " + word + " = " + str(int(number)))
    
    for phrase, number in {'thousands of': 10**3, 'millions of': 10**6, 'billions of': 10**9, 'trillions of': 10**12}.items():
        if phrase in text:
            number_information.append(phrase + " = " + str(int(number)))
    
    large_number_evidence = ""
    for info in number_information:
        large_number_evidence += info + "; "
    
    return large_number_evidence.strip()

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

def remove_similar_comments(names, comments):
    '''
    Remove table (or column) comments that have a high degree of similarity with their names
    
    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments
    
    Returns:
        new_comments: a list of new table (or column) comments
    '''
    new_comments = []
    for name, comment in zip(names, comments):    
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)
    
    return new_comments

def str_replace_ignore_case(evidence, schema_item_name):
    evidence = re.sub(re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE)

    return evidence

def obtain_n_grams(sequence, max_n):
    '''
    returns all grams of sequence less than or equal to `max_n`
    '''
    tokens = word_tokenize(sequence)
    all_grams = []
    for n in range(1, max_n + 1):
        all_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    
    return all_grams

def preprocess_evidence(evidence, schema_items):
    if evidence.strip() == "":
        return ""

    evidence = evidence.strip()
    # if evidence does not end with ";", add a ";" char
    if not evidence.endswith(";"):
        evidence += ";"
    
    # lowercase schema items appeared in the evidence
    for table in schema_items:
        if table["table_name"] in evidence.lower():
            evidence = str_replace_ignore_case(evidence, table["table_name"])

        for column_name in table["column_names"]:
            if column_name in evidence.lower():
                evidence = str_replace_ignore_case(evidence, column_name)
    
    evidence = evidence.replace("< =", "<=").replace("> =", ">=")

    return evidence

import os
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
import sqlparse
# from moz_sql_parser import Parser  # Assuming you're using moz_sql_parser

def process_data(data, db_path, db_comments, db_content_index_api, source, use_evidence, mode):
    sample = {}
    db_id = data["db_id"]
    
    sample["source"] = source
    sample["db_id"] = db_id
    sample["db_path"] = os.path.join(db_path, db_id, db_id + ".sqlite")

    if "spider-syn" in source:
        sample["question"] = data["SpiderSynQuestion"]
        sample["evidence"] = ""
    elif "bird" in source:
        sample["question"] = data["question"]
    elif "bank" in source:
        sample["question"] = data["question"]
        sample["evidence"] = extract_large_numbers(data["question"])
    else:
        sample["question"] = data["question"]
        sample["evidence"] = ""
    
    if "\n" in sample["question"]:
        sample["question"] = sample["question"].replace("\n", " ")
    
    sample["schema"] = get_db_schema(db_content_index_api, source, sample['question'], sample["db_path"], db_comments, db_id)
    if "bird" in source:
        evidence = preprocess_evidence(data["evidence"], sample["schema"]["schema_items"])
        sample["evidence"] = evidence

    if "\n" in sample["evidence"]:
        sample["evidence"] = sample["evidence"].replace("\n", " ")

    sample["text"] = sample["evidence"] + " " + sample["question"] \
        if use_evidence and sample["evidence"] != "" else sample["question"]

    if mode in ["train", "dev"]:
        sql = data["SQL"] if source in ["bird-dev", "bird-train"] else data["query"]
        sample['sql'] = sql
        # sample["sql"] = remove_table_alias(sqlparse.format(sql, keyword_case="upper", identifier_case="lower"))
    elif mode == "test":
        sample["sql"] = ""
    
    sample["table_labels"], sample["column_labels"] = [], []
    try:
        sql_tokens = [token.value for token in Parser(sample["sql"].lower()).tokens]
    except Exception as e:
        sql_tokens = sample["sql"].lower().split()
    
    for table_info in sample["schema"]["schema_items"]:
        if mode in ["train", "dev"]:
            table_name = table_info["table_name"]
            sample["table_labels"].append(1 if table_name in sql_tokens else 0)
            sample["column_labels"].append([
                1 if column_name in sql_tokens or f"{table_name}.{column_name}" in sql_tokens else 0
                for column_name in table_info["column_names"]
            ])
        elif mode == "test":
            sample["table_labels"].append(0)
            sample["column_labels"].append([0 for _ in range(len(table_info["column_names"]))])

    # Coarse-grained matching between the input text and all contents in the database

    
    return sample

def process_data_wrapper(args):
    return process_data(*args)

def spider_style_dataset(
    dataset_path, 
    db_path, 
    db_content_index_api, 
    source, 
    table_json_path,
    use_evidence,
    mode,
    output_file
):
    '''
    Load spider-style dataset
    
    Arguments:
        dataset_path: directory to load the dataset from
        db_path: directory of databases (used for extracting schema, including tables, columns, column contents, and foreign keys)
        db_content_index_path: directory of database content sparse index
        source: source of examples
        table_json_path: directory to load additional database information (used for extracting comments for tables and columns)
        use_evidence: whether to use the additional evidence in the input sequence
    Returns:
        returned_dataset: prepared dataset
    '''
    dataset = json.load(open(dataset_path))
    additional_db_info = json.load(open(table_json_path))

    # load old results from output_file if it exists
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_dataset = [json.loads(line) for line in f]
        processed_dataset_dict = {f"{sample['db_id']} {sample['question']}": sample for sample in processed_dataset}
    else:
        processed_dataset_dict = dict()
    
    # filter out processed samples
    dataset = [data for data in dataset if f"{data['db_id']} {data.get('question', '')}" not in processed_dataset_dict]

    db_comments = dict()
    # record comments for tables and columns
    for db_info in additional_db_info:
        comment_dict = dict()

        column_names = [column_name.lower() for _, column_name in db_info["column_names_original"]]
        table_idx_of_each_column = [t_idx for t_idx, _ in db_info["column_names_original"]]
        column_comments = [column_comment.lower() for _, column_comment in db_info["column_names"]]
        
        assert len(column_names) == len(column_comments)
        column_comments = remove_similar_comments(column_names, column_comments)

        table_names = [table_name.lower() for table_name in db_info["table_names_original"]]
        table_comments = [table_comment.lower() for table_comment in db_info["table_names"]]
        
        assert len(table_names) == len(table_comments)
        table_comments = remove_similar_comments(table_names, table_comments)

        # enumerate each table and its columns
        for table_idx, (table_name, table_comment) in enumerate(zip(table_names, table_comments)):
            comment_dict[table_name] = {
                "table_comment": table_comment,
                "column_comments": dict()
            }
            for t_idx, column_name, column_comment in zip(table_idx_of_each_column, column_names, column_comments):
                # record columns in current table
                if t_idx == table_idx:
                    comment_dict[table_name]["column_comments"][column_name] = column_comment

        db_comments[db_info["db_id"]] = comment_dict


    args_iter = zip(
        dataset,
        repeat(db_path),
        repeat(db_comments),
        repeat(db_content_index_api),
        repeat(source),
        repeat(use_evidence),
        repeat(mode)
    )


    pool =  Pool(processes=16)
    f_out = open(output_file, 'a+', encoding='utf-8')

    try:
        for sample in tqdm(
            pool.imap_unordered(process_data_wrapper, args_iter),
            total=len(dataset),
            desc="Processing dataset"
        ):
            # Write the JSON serialized sample to the file
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    except Exception as e:
        print(e)
        f_out.close()
        pool.close()
        import sys
        sys.exit()
    
    f_out.close()
    pool.close()

    # rearrange the dataset, load jsonl file, rearrange the dataset to correct order with the same order as the original dataset, key= {db_id + question}
    processed_dataset = []
    with open(output_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            sample = json.loads(line)
            processed_dataset.append(sample)

    dataset = json.load(open(dataset_path))
    rearranged_dataset = []
    for data in dataset:
        db_id = data["db_id"]
        if "spider-syn" in source:
            question = data["SpiderSynQuestion"]
        else:
            question = data["question"]

        key = db_id + " " + question.replace("\n", " ")
        for sample in processed_dataset:
            if sample["db_id"] + " " + sample["question"].replace("\n", " ") == key:
                rearranged_dataset.append(sample)
                break
    
    # save the rearranged dataset to json file, replace jsonl in output_file to json
    with open(output_file.replace(".jsonl", ".json"), 'w+', encoding='utf-8') as f_out:
        json.dump(rearranged_dataset, f_out, indent=2, ensure_ascii=False)

    return rearranged_dataset
    

if __name__ == "__main__":
    print("BIRD-dev (with evidence)")
    # BIRD dev set (1534 examples)
    bird_with_evidence_dev = spider_style_dataset(
        dataset_path = "./data/bird-062024/dev/dev.json", 
        db_path = "./data/bird-062024/dev/dev_databases", 
        db_content_index_api = "http://localhost:8005",
        source = "bird-dev",
        table_json_path = "./data/bird-062024/dev/dev_tables.json",
        use_evidence = True,
        mode = "dev",
        output_file="data/full_value_matching_sft_bird_062024_with_evidence_dev_text2sql.jsonl"
    )

    # print("BIRD (with evidence) train")
    # # BIRD training set with evidence (9428 examples)
    # bird_with_evidence_train = spider_style_dataset(
    #     dataset_path = "./data/bird-062024/train/train.json", 
    #     db_path = "./data/bird-062024/train/train_databases", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "bird-train",
    #     table_json_path = "./data/bird-062024/train/train_tables.json",
    #     use_evidence = True,
    #     mode = "train",
    #     output_file="data/full_value_matching_sft_bird_062024_with_evidence_train_text2sql.jsonl"
    # )


    # print("BIRD-dev (with evidence)")
    # bird_with_evidence_dev = spider_style_dataset(
    #     dataset_path = "data/sft_data_collections/bird/dev/dev.json", 
    #     db_path = "data/sft_data_collections/bird/dev/dev_databases", 
    #     db_content_index_path = "data/sft_data_collections/bird/dev/db_contents_index",
    #     source = "bird-dev",
    #     table_json_path = "data/sft_data_collections/bird/dev/dev_tables.json",
    #     use_evidence = True,
    #     mode = "dev",
    #     output_file="./data/sft_bird_with_evidence_dev_text2sql.jsonl"
    # )

    # print("BIRD (with evidence) train")
    # # BIRD training set with evidence (9428 examples)
    # bird_with_evidence_train = spider_style_dataset(
    #     dataset_path = "data/sft_data_collections/bird/train/train.json", 
    #     db_path = "data/sft_data_collections/bird/train/train_databases", 
    #     db_content_index_path = "data/sft_data_collections/bird/train/db_contents_index",
    #     source = "bird-train",
    #     table_json_path = "data/sft_data_collections/bird/train/train_tables.json",
    #     use_evidence = True,
    #     mode = "train",
    #     output_file="./data/sft_bird_with_evidence_train_text2sql.jsonl"
    # )

    # print("preparing training sets.....")
    # print("spider-train")
    # spider_train = []
    # # Spider training set-1 (7000 + 1658 examples)
    # for spider_train_set in ["train_spider.json", "train_others.json"]:
    #     spider_train.extend(
    #         spider_style_dataset(
    #             dataset_path = os.path.join("./data/sft_data_collections/spider/", spider_train_set), 
    #             db_path = "./data/sft_data_collections/spider/database", 
    #             db_content_index_path = "./data/sft_data_collections/spider/db_contents_index",
    #             source = "spider-train",
    #             table_json_path = "./data/sft_data_collections/spider/tables.json",
    #             use_evidence = False,
    #             mode = "train",
    #             output_file=f"./data/sft_spider_train_text2sql_{spider_train_set}.jsonl"
    #         )
    #     )
    # with open("./data/sft_spider_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(spider_train, indent = 2, ensure_ascii = False))
    # print("preparing training sets.....")
    # print("spider-train")
    # spider_train = []
    # # Spider training set-1 (7000 + 1658 examples)
    # for spider_train_set in ["train_spider.json", "train_others.json"]:
    #     spider_train.extend(
    #         spider_style_dataset(
    #             dataset_path = os.path.join("./data/sft_data_collections/spider/", spider_train_set), 
    #             db_path = "./data/sft_data_collections/spider/database", 
    #             db_content_index_api = "http://localhost:8005",
    #             source = "spider-train",
    #             table_json_path = "./data/sft_data_collections/spider/tables_update.json",
    #             use_evidence = False,
    #             mode = "train",
    #             output_file=f"./data/sft_spider_train_with_meaning_text2sql_{spider_train_set}.jsonl"
    #         )
    #     )
    # with open("./data/sft_spider_train_with_meaning_text2sql.json", "w") as f:
    #     f.write(json.dumps(spider_train, indent = 2, ensure_ascii = False))

    # print("preparing training sets.....")
    # print("spider-train-augmented")
    # spider_train = []
    # spider_dev = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/spider/train_augmented.json", 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "spider-train",
    #     table_json_path = "./data/sft_data_collections/spider/tables.json",
    #     use_evidence = False,
    #     mode = "train",
    #     output_file='./data/sft_spider_train_augmented_text2sql.jsonl'
    # )

    # print("BIRD (without evidence) train")
    # # BIRD training set (9428 examples)
    # bird_train = spider_style_dataset(
    #     dataset_path = "./data/bird-062024/train/train.json", 
    #     db_path = "./data/bird-062024/train/train_databases", 
    #     db_content_index_path = "./data/bird-062024/train/db_contents_index",
    #     source = "bird-train",
    #     table_json_path = "./data/bird-062024/train/train_tables.json",
    #     use_evidence = False,
    #     mode = "train"
    # )
    # with open("./data/sft_bird_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(bird_train, indent = 2, ensure_ascii = False))

    
    # with open("./data/sft_bird_with_evidence_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(bird_with_evidence_train, indent = 2, ensure_ascii = False))
    
    
    # print("---------------------------------------------------------------------------")
    # print("preparing dev sets.....")
    # print("spider-dev")
    # # Spider development set (1034 examples)
    # spider_dev = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/spider/dev.json", 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "spider-dev",
    #     table_json_path = "./data/sft_data_collections/spider/tables.json",
    #     use_evidence = False,
    #     mode = "dev",
    #     output_file='./data/1_value_sft_spider_dev_text2sql.jsonl'
    # )

    # print("---------------------------------------------------------------------------")
    # print("preparing dev sets.....")
    # print("spider-dev")
    # # Spider development set (1034 examples)
    # spider_dev = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/spider/dev.json", 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "spider-dev",
    #     table_json_path = "./data/sft_data_collections/spider/tables_update.json",
    #     use_evidence = False,
    #     mode = "dev",
    #     output_file='./data/sft_spider_dev_with_meaning_text2sql.jsonl'
    # )


    # print("spider-dk")
    # # Spider-DK development set (535 examples)
    # spider_dk = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/Spider-DK/Spider-DK.json", 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #     db_content_index_api = f"http://localhost:8005",
    #     source = "spider-dk",
    #     table_json_path = "./data/sft_data_collections/Spider-DK/tables.json",
    #     use_evidence = False,
    #     mode = "dev",
    #     output_file='./data/1_value_sft_spider_dk_text2sql.jsonl'
    # )
    
    # print("spider-syn")
    # # Spider-Syn development set (1034 examples)
    # spider_syn = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/Spider-Syn/Spider-Syn/dev.json", 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #     db_content_index_api = f"http://localhost:8005",
    #     source = "spider-syn-dev",
    #     table_json_path = "./data/sft_data_collections/spider/tables.json",
    #     use_evidence = False,
    #     mode = "dev",
    #     output_file='./data/1_value_sft_spider_syn_text2sql.jsonl'
    # )
    
    # print("spider-realistic")
    # # Spider-Realistic development set (507 examples)
    # spider_realistic = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/spider-realistic/spider-realistic.json", 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #    db_content_index_api = f"http://localhost:8005",
    #     source = "spider-realistic",
    #     table_json_path = "./data/sft_data_collections/spider/tables.json",
    #     use_evidence = False,
    #     mode = "dev",
    #     output_file='./data/1_value_sft_spider_realistic_text2sql.jsonl'
    # )

    # import signal
    # print("DR.spider")
    # dr_spider = []
    # # Dr.Spider has 17 perturbation test sets
    # test_set_names = os.listdir("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data")
    # test_set_names.remove("Spider-dev")
    # port = 8005
    # for test_set_name in test_set_names:
    #     if test_set_name.startswith("DB_"):
    #         database_file_path = "database_post_perturbation"
    #         table_file_name = "tables_post_perturbation.json"
    #     else:
    #         database_file_path = "databases"
    #         table_file_name = "tables.json"

    #     source = "dr.spider-{}".format(test_set_name)
    #     # run db content retrieval for each test set
    #     process = subprocess.Popen(f"python db_content_retrieval/lsh_api.py --port {port} --db_content_index {source}", shell=True)
    #     pid = process.pid
    #     # os.system(f"python db_content_retrieval/lsh_api.py --db_content_index {source}")
    #     import time
    #     time.sleep(10)
    #     dr_spider.extend(
    #             spider_style_dataset(
    #             dataset_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "questions_post_perturbation.json"), 
    #             db_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, database_file_path), 
    #             db_content_index_api = f"http://localhost:{port}",
    #             source = source,
    #             table_json_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, table_file_name),
    #             use_evidence = False,
    #             mode = "dev",
    #             output_file=f'./data/sft_dr_spider_text2sql_{test_set_name}.jsonl'
    #         )
    #     )
    #     # kill db content retrieval server
    #     # os.kill(pid, signal.SIGTERM)   # usually kills processes
    #     # os.kill(pid, signal.SIGKILL)   # should always kill a process
    #     os.system(f"kill -9 `ps aux | grep lsh_api.py | awk '{{print $2}}'`")
    #     time.sleep(2)
    # with open("./data/sft_dr_spider_text2sql.json", "w") as f:
    #     f.write(json.dumps(dr_spider, indent = 2, ensure_ascii = False))


    # print("BIRD-dev (without evidence)")
    # # BIRD dev set (1534 examples)
    # bird_dev = spider_style_dataset(
    #     dataset_path = "./data/bird-062024/dev/dev.json", 
    #     db_path = "./data/bird-062024/dev/dev_databases", 
    #     db_content_index_path = "./data/bird-062024/dev/db_contents_index",
    #     source = "bird-dev",
    #     table_json_path = "./data/bird-062024/dev/dev_tables.json",
    #     use_evidence = False,
    #     mode = "dev"
    # )
    # with open("./data/sft_bird_dev_text2sql.json", "w") as f:
    #     f.write(json.dumps(bird_dev, indent = 2, ensure_ascii = False))

    
    # print("Bank_Financials dev set")
    # # Bank_Financials dev set (92 examples)
    # bank_dev = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/domain_datasets/Bank_Financials_dev.json", 
    #     db_path = "./data/sft_data_collections/domain_datasets/databases", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "bank_financials-dev",
    #     table_json_path = "./data/sft_data_collections/domain_datasets/tables.json",
    #     use_evidence = True,
    #     mode = "dev",
    #     output_file="./data/sft_bank_financials_dev_text2sql.jsonl"
    # )
    
    # print("Aminer_Simplified dev set")
    # # Aminer_Simplified dev set (xxx examples)
    # aminer_dev = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/domain_datasets/Aminer_Simplified_dev.json", 
    #     db_path = "./data/sft_data_collections/domain_datasets/databases", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "aminer_simplified-dev",
    #     table_json_path = "./data/sft_data_collections/domain_datasets/tables.json",
    #     use_evidence = True,
    #     mode = "dev",
    #     output_file="./data/sft_aminer_simplified_dev_text2sql.jsonl"
    # )


    # print("Bank_Financials train")
    # # Bank_Financials train set
    # bank_train = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/domain_datasets/Bank_Financials_train.json", 
    #     db_path = "./data/sft_data_collections/domain_datasets/databases", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "bank_financials-train",
    #     table_json_path = "./data/sft_data_collections/domain_datasets/tables.json",
    #     use_evidence = True,
    #     mode = "train",
    #     output_file="./data/sft_bank_financials_train_text2sql.jsonl"
    # )

    # print("Aminer_Simplified train")
    # # Aminer_Simplified train set
    # aminer_train = spider_style_dataset(
    #     dataset_path = "./data/sft_data_collections/domain_datasets/Aminer_Simplified_train.json", 
    #     db_path = "./data/sft_data_collections/domain_datasets/databases", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "aminer_simplified-train",
    #     table_json_path = "./data/sft_data_collections/domain_datasets/tables.json",
    #     use_evidence = True,
    #     mode = "train",
    #     output_file="./data/sft_aminer_simplified_train_text2sql.jsonl"
    # )
    
    # print("Spider + BIRD + Bank_Financials + Aminer_Simplified train set (ALL MERGED)")
    # # merge all available training data
    # with open("./data/sft_all_merged_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(spider_train + bird_with_evidence_train + bank_train + aminer_train, indent = 2, ensure_ascii = False))
    

    pass


    # Other un-official SFT files for testing
    # print("preparing training sets.....")
    # print("spider-train")
    # os.remove('./data/sft_spider_train_domain_geo.jsonl')
    # spider_style_dataset(
    #     dataset_path = './data/sft_data_collections/spider_domain_geo.json', 
    #     db_path = "./data/sft_data_collections/spider/database", 
    #     db_content_index_api = "http://localhost:8005",
    #     source = "spider-train",
    #     table_json_path = "./data/sft_data_collections/spider/tables.json",
    #     use_evidence = False,
    #     mode = "train",
    #     output_file=f"./data/sft_spider_train_domain_geo.jsonl"
    # )
