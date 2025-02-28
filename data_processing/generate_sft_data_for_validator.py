import argparse
import json
from datasets import Dataset, DatasetDict
import numpy as np
from planner import _make_str_response, _execute_sql, is_execution_correct
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from multiprocessing import Manager

# Set seed
np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../data/llm_alignment/validator_join_bird_with_evidence_train.jsonl')
parser.add_argument('--output_dir', type=str, default='../data/llm_alignment/sft_data_for_validator_join_bird_with_evidence_train')
args = parser.parse_args()

def norm_completion(completion):
    """Normalize the completion by removing unwanted lines."""
    lines = completion.split('\n')
    filter_lines = [line for line in lines if "broaden the criteria" not in line]
    completion = "\n".join(filter_lines)
    return completion

PROMPT = """Generate feedbacks to fix the following SQL query:
{schema}

Question: {question}
External knowledge: {evidence}

SQL query: {sql_query}

Execution response:
{execution_response}

Feedback:"""

def process_sample(sample, index, input_file):
    """Process a single sample from the dataset."""
    key, token = None, None
    if '_join' in input_file:
        key = 'validator_join'
        token = "JOIN."
    elif '_select' in input_file:
        key = 'validator_select'
        token = "SELECT."
    elif '_order' in input_file:
        key = 'validator_order'
        token = "ORDER BY."
    elif '_condition' in input_file:
        key = 'validator_condition'
        token = "CONDITION."

    feedback = sample.get(key)
    if not feedback:
        return None
    
    if type(feedback) == list:
        feedback = feedback[0]
    
    if f"prompt_{key}" in sample:
        prompt = sample[f"prompt_{key}"]
        prompt_completion = prompt + feedback
    else:
        prompt_completion = "\n" + feedback

    feedback = token + prompt_completion.split(token)[-1]
    prompt = PROMPT.format(schema=sample['schema_sequence'], 
                           question=sample['question'], 
                           evidence=sample['evidence'],
                           sql_query=sample['predict_sql'], 
                           execution_response=sample['pred_result'])

    completion = feedback
    if isinstance(completion, list):
        completion = completion[0]

    completion = norm_completion(completion)
    prompt_id = f"{index}"

    true_result, _ = _execute_sql("./" + sample["db_path"], sample["sql"])
    pred_result, _ = _execute_sql("./" + sample["db_path"], sample['predict_sql'])

    is_pred_sql_correct = is_execution_correct(true_result, pred_result)
    feedback_conclude_correct = completion is None or 'Conclude: correct' in completion

    if is_pred_sql_correct and not feedback_conclude_correct:  # bad case
        return None

    return {
        'prompt_id': prompt_id,
        'messages': {
            'prompt': prompt,
            'completion': completion
        }
    }

def main():
    # Load JSONL data
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Use multiprocessing to process samples with tqdm
    print('Start processing samples...')
    with Manager() as manager:
        results_list = manager.list()  # Shared list for results
        total_samples = len(data)

        with tqdm(total=total_samples, desc="Processing Samples") as pbar:
            def update_progress(result):
                if result is not None:
                    results_list.append(result)
                pbar.update()

            with Pool(processes=24) as pool:
                partial_process_sample = partial(process_sample, input_file=args.input_file)
                for idx, sample in enumerate(data):
                    pool.apply_async(partial_process_sample, args=(sample, idx), callback=update_progress)

                pool.close()
                pool.join()

        # Convert results to a normal list
        sft_data = list(results_list)

    # Shuffle and create DatasetDict
    np.random.shuffle(sft_data)

    dataset = DatasetDict({
        'train': Dataset.from_list(sft_data),
        'test': Dataset.from_list(sft_data[:100]),
    })

    print(dataset)
    dataset.save_to_disk(args.output_dir)

if __name__ == "__main__":
    main()
