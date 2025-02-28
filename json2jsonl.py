import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json-file', type=str, required=True)
args = parser.parse_args()

file = args.json_file

# Read the JSON file
with open(file, 'r') as f:
    data = json.load(f)

# Export to JSONL
with open(file.replace('.json', '.jsonl'), 'w') as f:
    for record in data:
        f.write(json.dumps(record) + '\n')
