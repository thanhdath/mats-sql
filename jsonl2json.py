import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--jsonl-file', type=str, required=True)
args = parser.parse_args()

file = args.jsonl_file
data = []
with open(file, 'r') as f:
    for line in f:
        data.append(json.loads(line))
# export to json, replace jsonl to json
with open(file.replace('jsonl', 'json'), 'w') as f:
    json.dump(data, f, indent=4)
    