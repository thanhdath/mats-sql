from datasets import load_from_disk
import argparse
import datasets
import numpy as np
from datasets import Dataset

# add arguments data/llm_alignment/spider-p1 
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="path to the data directory")
args = parser.parse_args()

# read all train data from the data directory, including
# dpo-llama-3-end2end-spider_train_fixed_sql
# dpo-llama-3-end2end-spider_train_planner
# dpo-llama-3-end2end-spider_train_validator_condition
# dpo-llama-3-end2end-spider_train_validator_join
# dpo-llama-3-end2end-spider_train_validator_select
# dpo-llama-3-end2end-spider_train_validator_order

import glob
import os
data_dirs = glob.glob(args.data_dir + "/*train*")
data_dirs = [x for x in data_dirs if os.path.isdir(x)]
print(data_dirs)

for data_dir in data_dirs:
    dataset_train = load_from_disk(data_dir)
    # load dev data
    dev_file = data_dir.replace("train", "dev")
    if os.path.exists(dev_file):
        dataset_dev = load_from_disk(dev_file)
        dataset_dev = list(dataset_dev['train_dpo'])
        dataset_dev = np.random.permutation(dataset_dev)[:2000].tolist()
        dataset_train['test_dpo'] = Dataset.from_list(dataset_dev)

    print(data_dir)
    print(dataset_train)

    # save the merged data to other directory
    dataset_train.save_to_disk(data_dir.replace("train", "train_dev"))
    print(data_dir.replace("train", "train_dev"))
