import argparse
import os
import torch
import json
import time
from tqdm import tqdm

from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import detect_special_char
from validator import Validator
from sql_agent import SQLAgent

class SQLGenerator():
    def __init__(self, sql_llm_path, val_llm_path):
        # load model
        self.validator = Validator(val_llm_path, llm_path=val_llm_path, api_base=None)
        if val_llm_path == sql_llm_path:
            self.sql_agent = SQLAgent(None)
            self.sql_agent.model = self.validator.model
            self.sql_agent.tokenizer = self.validator.tokenizer
        else:
            self.sql_agent = SQLAgent(sql_llm_path)

        self.model = self.validator.model
        self.tokenizer = self.validator.tokenizer

    def text2sql(self, data, 
                    max_new_tokens, 
                    num_beams=4, 
                    num_return_sequences=4, 
                    do_sample=False,
                    temperature=0.0,
                    n_turns=3):
        print("-"*50)
        print("Question:", data["question"])
        print("True SQL:", data["sql"])
        
        self.sql_agent.reset(data)
        n_turn = 0
        all_message_feedbacks = []

        while n_turn <= n_turns:
            generated_sqls = self.sql_agent.generate_sql(
                max_new_tokens, 
                num_beams, 
                num_return_sequences,
                do_sample=do_sample,
                temperature=temperature
                )
            if len(generated_sqls) == 0:
                break

            generated_sqls = list(set(generated_sqls))
            self.sql_agent.pick_best_sql(generated_sqls)
            print('\n'.join([f"{i}: {generated_sql}" for i, generated_sql in enumerate(generated_sqls)]))

            # get feedback from validator
            str_schema = f"""{data["schema_sequence"]}
{data["content_sequence"]}"""
            
            for generated_sql in generated_sqls:
                feedbacks, message_feedbacks = self.validator.get_answer(schema=str_schema, 
                                                question=data["question"],
                                                evidence=data["evidence"],
                                                sql_query=generated_sql,
                                                db_path=data["db_path"],
                                                do_sample=do_sample,
                                                temperature=temperature,
                                                num_return_sequences=num_return_sequences)
                # print('\n'.join([f"{i}: {message_feedback[-1]['content']}" for i, message_feedback in enumerate(message_feedbacks)]))
                all_message_feedbacks.extend(message_feedbacks)

            feedback = feedbacks[0]
            if "Correct SQL" in feedback:
                break

            # update message
            self.sql_agent.receive_feedback(feedback)
                            
        return all_message_feedbacks
        

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql_llm_path', type = str)
    parser.add_argument('--val_llm_path', type = str)
    parser.add_argument('--sic_path', type = str)
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)

    parser.add_argument('--dataset_path', type = str)

    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)
    parser.add_argument('--n_turns', type = int, default = 3)

    parser.add_argument('--output_file', type = str, default = "log.json")
    
    opt = parser.parse_args()

    return opt

def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql

if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    sql_generator = SQLGenerator(opt.sql_llm_path, opt.val_llm_path)
    tokenizer = sql_generator.tokenizer
    
    eval_set = SFTSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        max_tokens - max_new_tokens,
        "eval",
        opt.table_num,
        opt.column_num,
        opt.sic_path,
        do_filter_schema = False    
    )

    # TODO: current, we only support batch size = 1
    # dataloader = DataLoader(eval_set, batch_size = 1)
    os.makedirs(os.path.dirname(opt.output_file), exist_ok = True)

    start_time = time.time()
    predicted_sqls = []

    if os.path.isfile(opt.output_file):
        all_feedback_messages = json.load(open(opt.output_file))
    else:
        all_feedback_messages = []


    for idata in tqdm(range(len(all_feedback_messages), len(eval_set.dataset))):
        data = eval_set.dataset[idata]
        message_feedbacks = sql_generator.text2sql(
            data, 
            max_new_tokens, 
            num_beams=1, 
            num_return_sequences=3, 
            do_sample=True,
            temperature=0.9,
            n_turns=opt.n_turns
        )
        all_feedback_messages.append(message_feedbacks)

        if idata % 10 == 0:
            json.dump(all_feedback_messages, open(opt.output_file, "w"), indent = 2)

    end_time = time.time()
    print("LLM name: {} - {} | Total time: {}s | Example number: {} | Average time: {}s".format(
        opt.sql_llm_path, 
        opt.val_llm_path, 
        end_time - start_time,
        len(eval_set.dataset),
        (end_time - start_time) / len(eval_set.dataset)
        )
    )
    json.dump(all_feedback_messages, open(opt.output_file, "w"), indent = 2)
