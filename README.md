[MATS: Collaborative Multi-agent Local Small Language Models for Text2SQL with Execution Reinforcement]


**1. To set up tbe environment**
```
conda env create -n mats -f environment.yml
conda activate mats
```

**2. Run Evaluation on BIRD**:

First serve the models with VLLM.
```
CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-3b-bird-planner --host 0.0.0.0 --port 8003 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name planner --gpu-memory-utilization 0.3 --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-1b-bird-validator --host 0.0.0.0 --port 8004 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name validator --gpu-memory-utilization 0.2  --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-1b-bird-fixed --host 0.0.0.0 --port 8005 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name fixed --gpu-memory-utilization 0.2  --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-3b-bird-selection --host 0.0.0.0 --port 8006 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name selection --gpu-memory-utilization 0.3  --enable-prefix-caching
```

Run evaluation:
```
eval_file=data/evaluate/orpo-llama-3-iter-2-end2end-bird_dev.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/schema_insight_bird_with_evidence_dev_text2sql.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --n_processes 16

python compute_acc.py --pred_file $eval_file
```




**3. To run evaluation on Spider**:

First serve the models with VLLM.
```
CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-3b-spider-planner --host 0.0.0.0 --port 8003 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name planner --gpu-memory-utilization 0.3 --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-1b-spider-validator --host 0.0.0.0 --port 8004 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name validator --gpu-memory-utilization 0.2  --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-1b-spider-fixed --host 0.0.0.0 --port 8005 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name fixed --gpu-memory-utilization 0.2  --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-3b-spider-selection --host 0.0.0.0 --port 8006 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name selection --gpu-memory-utilization 0.3  --enable-prefix-caching
```

Run evaluation:
```
eval_file=data/evaluate/orpo-llama-3-iter-2-end2end-spider_dev.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/schema_insight_spider_dev_text2sql.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --n_processes 16

python compute_acc.py --pred_file $eval_file
```


**4. For training agents**

The Schema Filtering is inherited from CodeS (https://github.com/RUCKBReasoning/codes).

To train other agents, see the code in ***alignment-handbook/***, here we modified the repository alignment-handbook (https://github.com/huggingface/alignment-handbook) for supervised-finetuning and ORPO on the completion part only. The config files could be found in **alignment-handbook/recipes/**.

**Note**: Currently this work is under review. The model and training dataset will be publicly available upon acceptance.
