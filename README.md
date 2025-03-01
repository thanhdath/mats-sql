[MATS: Collaborative Multi-agent Local Small Language Models for Text2SQL with Execution Reinforcement]


**Set up environment**
```
conda env create -n mats -f environment.yml
conda activate mats
```

**Run Evaluation on BIRD**:
First serve the models with VLLM.
```
CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-3b-bird-planner --host 0.0.0.0 --port 8003 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name planner --gpu-memory-utilization 0.3 --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-1b-bird-validator --host 0.0.0.0 --port 8004 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name validator --gpu-memory-utilization 0.2  --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-1b-bird-fixed --host 0.0.0.0 --port 8005 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name fixed --gpu-memory-utilization 0.2  --enable-prefix-caching

CUDA_VISIBLE_DEVICES=0 vllm serve thanhdathoang/llama-3b-bird-selection --host 0.0.0.0 --port 8006 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name selection --gpu-memory-utilization 0.3  --enable-prefix-caching
```

Run evaluation:
```
eval_file=data/evaluate/orpo-llama-3-iter-2-planner-bird_dev-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/schema_insight_bird_with_evidence_dev_text2sql.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --n_processes 16

python compute_acc.py --pred_file $eval_file
```


**Note**: Currently this work is under review. The model and training dataset will be publicly available upon acceptance.
