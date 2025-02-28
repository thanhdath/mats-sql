import json
import os
from tqdm import tqdm
from planner import PlannerCombine, PlannerCombineWithTrueSQL
import argparse
from multiprocessing import Pool

# add parse for input data file (train, dev) and output_file
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../data/sft_bird_with_evidence_train_text2sql.json')
parser.add_argument('--output_file', type=str, default='../data/planner/planner_select_bird_with_evidence_train.jsonl')
parser.add_argument('--endpoint_type', type=str, default='llamacpp', choices=['vllm', 'llamacpp', 'openai'])
parser.add_argument('--mode', type=str, choices=['select', 'condition', 'combine', 'combine_with_true_sql'], default='combine')
parser.add_argument('--prompt', choices=['few-shot', 'cot'], default='few-shot')
args = parser.parse_args()

if args.input_file.endswith('.json'):
    data = json.load(open(args.input_file))
elif args.input_file.endswith('.jsonl'):
    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

# load saved output file if exists
if os.path.exists(args.output_file):
    old_output = []
    with open(args.output_file, 'r') as f:
        for line in f:
            old_output.append(json.loads(line))
    data[:len(old_output)] = old_output
else:
    old_output = []

# open jsonl file for append contents
# makedirs if not exists
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
output_file = open(args.output_file, 'a+')

if args.mode == 'combine':
    planner = PlannerCombine(endpoint_type=args.endpoint_type)
elif args.mode == 'combine_with_true_sql':
    planner = PlannerCombineWithTrueSQL(endpoint_type=args.endpoint_type)

if args.prompt == 'cot':
    planner.prompt_template = """{schema}

Question: {question}
External knowledge: {evidence}

Use this hidden True SQL query to write correct analysis that derives to the correct answer. The True SQL query cannot be used in the analysis.
Hidden True SQL query: {true_sql_query} 

Write your thought in short then write the final SQL query, answer in this format:
[your short thought step-by-step]
Final SQL query:
```
[SQL query]
```
"""

def process_sample(sample):
    answer = planner.generate(sample)
    sample[f'planner_{args.mode}'] = answer
    return sample

def main():
    chunk_size = 4
    with open(args.output_file, 'a') as output_file:
        for i in tqdm(range(len(old_output), len(data), chunk_size), total=(len(data) - len(old_output))//chunk_size):
            chunk = data[i:i+chunk_size]
            pool = Pool(chunk_size)
            processed_samples = pool.map(process_sample, chunk)
            pool.close()

            if len(processed_samples) > 0:
                print(processed_samples[0][f'planner_{args.mode}'])
            
            for sample in processed_samples:
                output_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
