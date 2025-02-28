import json
from torch.utils.data import DataLoader
import torch
import re
import numpy as np
from trl import PPOTrainer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig
from trl import PPOConfig
import argparse
from data_processing.planner import _make_str_response, _execute_sql, is_execution_correct
from data_processing.planner import Planner
from datasets import load_dataset, load_from_disk
from transformers import StoppingCriteria

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence):
        self.target_sequence = target_sequence

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = tokenizer.decode(input_ids[0])
        # Check if the target sequence appears in the generated text
        if generated_text.count(self.target_sequence) == 2:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

def extract_sql(plan):
    pred_sql_match = re.search(r'Final SQL query:\s*```(.*?)```', plan, re.DOTALL)
    if pred_sql_match is None:
        return ''
    pred_sql = pred_sql_match.group(1).replace("sql", "").replace("```", "").strip()
    return pred_sql

np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)


parser = argparse.ArgumentParser()
parser.add_argument("--model-base", default="alignment-handbook/output/llama-3b-bird-planner-fft")
parser.add_argument("--dataset", default='data/llm_alignment/bird-p1/dpo-llama-3-end2end-bird_train_planner.jsonl')
parser.add_argument("--save-iterations", default=20, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--mini-batch-size", default=1, type=int)
args = parser.parse_args()

device = "cuda:0"

if "codes-1b" in args.model_base:
    target_modules = [
        "c_proj",
        "c_attn",
        "c_fc"
    ]
elif "codes-3b" in args.model_base:
    target_modules = [
        "c_proj",
        "c_fc",
        "c_attn"
    ]
else:
    target_modules = 'all-linear'

batch_size=args.batch_size
mini_batch_size=args.mini_batch_size
gradient_accumulation_steps=batch_size // mini_batch_size
config = PPOConfig(
    model_name=args.model_base,
    learning_rate=5.0e-6,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    log_with="tensorboard",
    project_kwargs={"logging_dir": "log-tensorboard/sql"},
   # kl_penalty="full",
 #   adap_kl_ctrl=False,
 #   init_kl_coef=0.1
)

lora_config_sql = LoraConfig(
    target_modules=target_modules,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model_original = AutoModelForCausalLM.from_pretrained(
    args.model_base,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    trust_remote_code=True,
    device_map="auto")


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_original,
    # peft_config=lora_config_sql,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
# tokenizer.pad_token = tokenizer.eos_token
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    ref_model=None,
    tokenizer=tokenizer)

def get_first_turn_message(sample):
    messages = sample['messages']
    # get 1 turn without assistant message
    messages = [x for x in messages if x['role'] != 'assistant']
    sample['messages'] = messages
    return sample

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataset = []
with open(args.dataset) as fp:
    for line in fp:
        samples = json.loads(line)
        if len(samples) == 0:
            continue
        sample = samples[0]
        prompt = sample['prompt']
        # prompt = prompt.replace("<|start_header_id|>user<|end_header_id|>", "<|user|>")
        # prompt = prompt.replace("<|start_header_id|>assistant<|end_header_id|>", "<|assistant|>")
        # prompt = prompt.replace("<|eot_id|>", "<|end|>")
        db_path = sample['db_path']
        true_sql = extract_sql(sample['chosen'][0])
        dataset.append({
            'prompt': prompt,
            'db_path': db_path,
            'sql': true_sql
        })
dataset = dataset[:-100]
generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": 768,
    # "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "temperature": 0.8,
    # "eos_token_id": tokenizer.convert_tokens_to_ids(['<|end|>'])[0],
    "pad_token_id": tokenizer.eos_token_id,
    "stopping_criteria": MyStoppingCriteria("<|end|>")
}

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, 
                        num_workers=16, pin_memory=True, drop_last=True)

EOS_TOKEN = '<|eot_id|>'
ASSISTANT_TOKEN = '<|start_header_id|>assistant<|end_header_id|>'
USER_TOKEN = '<|start_header_id|>user<|end_header_id|>'
planner = Planner(prompt_file='data_processing/prompts/zero_shot_prompt_planner.txt', 
                    endpoint_type='vllm')
planner.prompt_template = USER_TOKEN + """
{schema}

Question: {question}
External knowledge: {evidence}

Planning:
""" + EOS_TOKEN + "\n" + ASSISTANT_TOKEN

# def generate(sample):
#     prompt = sample['prompt']
#     query_tensors = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
#     response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs)[0]

#     answer = tokenizer.decode(response_tensors, skip_special_tokens=True)
#     generated_sql = extract_sql(answer)
#     return prompt, query_tensors, response_tensors, generated_sql

def generate(samples):
    prompts = samples['prompt']
    query_tensors = []
    response_tensors = []
    answers = []
    generated_sqls = []
    for prompt in prompts:
        query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
        response_tensor = ppo_trainer.generate(query_tensor, return_prompt=False, generate_ref_response=False, **generation_kwargs)[0]
        answer = tokenizer.decode(response_tensor, skip_special_tokens=True)
        generated_sql = extract_sql(answer)
        query_tensors.append(query_tensor)
        response_tensors.append(response_tensor)
        answers.append(answer)
        generated_sqls.append(generated_sql)

    return prompts, query_tensors, response_tensors, answers, generated_sqls
import multiprocessing as mp

# Function for parallel execution
def execute_sql_parallel(args):
    db_path, sql = args
    return _execute_sql(db_path, sql)

# Updated SQL execution with multiprocessing
def execute_with_multiprocessing(db_paths, sqls, num_workers=8):
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(execute_sql_parallel, zip(db_paths, sqls))
    return results

for epoch in range(10):
    train_feedback_samples = []
    train_sql_samples = []
    iteration = 0

    for iteration, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Generate SQL and feedback for this sample
        n_turn = 0
        sql_reward = None

        # Using multiprocessing for true SQL execution
        true_execution = execute_with_multiprocessing(data["db_path"], data["sql"], num_workers=8)

        # Generate predicted SQL
        prompts, query_tensors, response_tensors, answers, generated_sqls = generate(data)
        print(generated_sqls[0])

        # Using multiprocessing for predicted SQL execution
        pred_execution = execute_with_multiprocessing(data["db_path"], generated_sqls, num_workers=8)

        # Compute rewards
        # rewards = [float(is_execution_correct(true[0], pred[0])) for true, pred in zip(true_execution, pred_execution)]
        rewards = []
        for true, pred in zip(true_execution, pred_execution):
            if pred[1]:
                reward = -1.0
            else:
                reward = float(is_execution_correct(true[0], pred[0]))
            rewards.append(reward)
        rewards = [torch.tensor(reward) for reward in rewards]

        print(rewards)

        # PPO training step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(
            stats,
            {"query": prompts, "response": answers},
            rewards
        )

        # Save model at specified iterations
        if iteration % args.save_iterations == 0:
            ppo_trainer.save_pretrained(f"output/ppo-2agents-{epoch}/sql")
    ppo_trainer.save_pretrained(f"output/ppo-2agents-{epoch}/sql")
