
# CUDA_VISIBLE_DEVICES=0 vllm serve orpo-llama-3b-iter-2-bird-planner-no-filter/ --host 0.0.0.0 --port 8003 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name planner --gpu-memory-utilization 0.9 --enable-prefix-caching

# CUDA_VISIBLE_DEVICES=1 vllm serve llama-1b-bird-validator-fft/ --host 0.0.0.0 --port 8004 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name validator --gpu-memory-utilization 0.5  --enable-prefix-caching

# CUDA_VISIBLE_DEVICES=1 vllm serve llama-1b-bird-fixed-fft/ --host 0.0.0.0 --port 8005 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name fixed --gpu-memory-utilization 0.4  --enable-prefix-caching


mkdir logs
mkdir logs/bird-dev
# # Evaluate 
n=10
temperature=1.0
seed=100
eval_file=data/evaluate/orpo-llama-3-iter-2-planner-bird_dev-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/full_value_matching_schema_insight_bird_062024_with_evidence_dev_text2sql.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --only_planner --seed 100 --n_processes 32

python jsonl2json.py --jsonl-file data/evaluate/orpo-llama-3-iter-2-planner-bird_dev-greedy-and-sampling-seed$seed.jsonl

# Evaluate Selection
eval_file=data/evaluate/orpo-llama-3-iter-2-selection-bird_dev-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/evaluate/orpo-llama-3-iter-2-planner-bird_dev-greedy-and-sampling-seed$seed.json \
	--output_file $eval_file \
	--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 \
	--skip_planner --seed 100 --n_processes 8 --skip_validator_join

python compute_acc.py --pred_file $eval_file

