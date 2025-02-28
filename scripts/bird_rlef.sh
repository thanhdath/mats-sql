# This script is for reproducing the result

# GIVEN SFT models (4 epochs)
# llama-3b-planner-bird-fft
# llama-3b-validator-bird-fft
# llama-3b-fixed-bird-fft

mkdir data/llm_alignment/bird/

# =================================================================================================
# GEN ORPO data for planner
# =================================================================================================

CUDA_VISIBLE_DEVICES=0 vllm serve llama-3b-bird-planner-fft/ --host 0.0.0.0 --port 8003 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name planner --gpu-memory-utilization 0.8

train_file=data/llm_alignment/llama-3b-planner-bird_train.jsonl
rm $train_file
PYTHONPATH=. python evaluate_end2end.py \
    --input_file data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json \
    --output_file $train_file \
    --model-name llama --mode train \
    --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --only_planner

DATA_DIR=data/llm_alignment/bird-p1-planner/
rm -r $DATA_DIR/
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file data/llm_alignment/orpo-llama-3b-planner-bird_train_beam_and_temperature.jsonl \
    --gpt_planner_file  data/multi-agents/planner/gpt-planner_combine_with_true_sql_bird_062024_with_evidence_train_combine.jsonl \
    --output_planner_file $DATA_DIR/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file $DATA_DIR/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl \
    --n_select_chosens 2


# =================================================================================================
# GEN ORPO data for planner Iter 2
# =================================================================================================
train_file=data/llm_alignment/orpo-llama-3b-planner-bird_train_temperature.jsonl
rm $train_file
PYTHONPATH=. python evaluate_end2end.py \
    --input_file data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json \
    --output_file $train_file \
    --model-name llama --mode train \
    --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --only_planner
# PYTHONPATH=. python evaluate_end2end.py \
#     --input_file data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json \
#     --output_file $train_file \
#     --model-name llama --mode train \
#     --n_return 4 --use_beam_search --temperature 0.0 --api_host http://localhost:8003 --only_planner

# DATA_DIR=data/llm_alignment/bird-p2-planner/
# rm -r $DATA_DIR/
# PYTHONPATH=. python llm_alignment/build_rl_data.py \
#     --input_file data/llm_alignment/orpo-llama-3b-planner-bird_train_beam_and_temperature.jsonl \
#     --gpt_planner_file  data/multi-agents/planner/gpt-planner_combine_with_true_sql_bird_062024_with_evidence_train_combine.jsonl \
#     --output_planner_file $DATA_DIR/dpo-llama-3-end2end-bird_train_planner.jsonl \
#     --output_validator_select_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
#     --output_validator_condition_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
#     --output_validator_join_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
#     --output_validator_order_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
#     --output_fixed_sql_file $DATA_DIR/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl \
#     --n_select_chosens 2

DATA_DIR=data/llm_alignment/bird-p2-planner/
rm -r $DATA_DIR/
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file $train_file \
    --gpt_planner_file  data/multi-agents/planner/gpt-planner_combine_with_true_sql_bird_062024_with_evidence_train_combine.jsonl \
    --output_planner_file $DATA_DIR/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file $DATA_DIR/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl \
    --n_select_chosens 2


# ITER 3

vllm serve orpo-llama-3b-iter-2-bird-planner/ --host 0.0.0.0 --port 8103 --dtype auto --max-model-len 4096 --disable-log-requests --served-model-name planner --gpu-memory-utilization 0.9

train_file=data/llm_alignment/orpo-llama-3b-iter-2-planner-bird_train_temperature.jsonl
rm $train_file
PYTHONPATH=. python evaluate_end2end.py \
    --input_file data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json \
    --output_file $train_file \
    --model-name llama --mode train \
    --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --only_planner


DATA_DIR=data/llm_alignment/bird-p3-planner/
rm -r $DATA_DIR/
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file $train_file \
    --gpt_planner_file  data/multi-agents/planner/gpt-planner_combine_with_true_sql_bird_062024_with_evidence_train_combine.jsonl \
    --output_planner_file $DATA_DIR/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file $DATA_DIR/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl \
    --n_select_chosens 2
scp -r $DATA_DIR grif:~/codes/data/llm_alignment/



# GEN ORPO FOR FIX

# train_file=data/llm_alignment/orpo-llama-3b-iter-2-planner-bird_train_greedy.jsonl
# PYTHONPATH=. python evaluate_end2end.py \
#     --input_file data/full_value_matching_schema_insight_bird_062024_with_evidence_train_text2sql.json \
#     --output_file $train_file \
#     --model-name llama --mode train \
#     --n_return 1 --temperature 0.0 --api_host http://192.168.1.108:8003 --only_planner

python jsonl2json.py --jsonl-file data/llm_alignment/orpo-llama-3b-iter-2-planner-bird_train_greedy.jsonl

PYTHONPATH=. python evaluate_end2end.py \
    --input_file data/llm_alignment/orpo-llama-3b-iter-2-planner-bird_train_greedy.json \
    --output_file data/llm_alignment/orpo-llama-3b-iter-2-validator-bird_train_greedy.jsonl\
    --model-name llama --mode train \
    --n_return 1 --temperature 0.0 --api_host http://localhost:8003 --skip_planner --skip_fix --skip_validator_order

python jsonl2json.py --jsonl-file data/llm_alignment/orpo-llama-3b-iter-2-validator-bird_train_greedy.jsonl

PYTHONPATH=. python evaluate_end2end.py \
    --input_file data/llm_alignment/orpo-llama-3b-iter-2-validator-bird_train_greedy.json \
    --output_file data/llm_alignment/orpo-llama-3b-iter-2-fix-bird_train_sampling.jsonl\
    --model-name llama --mode train \
    --n_return 10 --temperature 1.0 --api_host http://localhost:8003 --skip_planner --skip_validator


DATA_DIR=data/llm_alignment/bird-p1-fix/
rm -r $DATA_DIR
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file data/llm_alignment/orpo-llama-3b-iter-2-fix-bird_train_sampling.jsonl \
    --output_planner_file $DATA_DIR/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file $DATA_DIR/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file $DATA_DIR/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl







# =================================================================================================
# GEN ORPO data for Validator
# =================================================================================================


CUDA_VISIBLE_DEVICES=0 vllm serve llama-3b-bird-validator-fft/ --host 0.0.0.0 --port 8004 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name validator --gpu-memory-utilization 0.8

CUDA_VISIBLE_DEVICES=1 vllm serve llama-3b-bird-fixed-fft-follow-validation/ --host 0.0.0.0 --port 8005 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name fixed --gpu-memory-utilization 0.8

python jsonl2json.py --jsonl-file data/llm_alignment/bird/llama-3b-planner-bird_train.jsonl
python jsonl2json.py --jsonl-file data/llm_alignment/bird/llama-3b-planner-bird_dev.jsonl

rm data/llm_alignment/bird/llama-3b-validator-bird_train.jsonl
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/llm_alignment/bird/llama-3b-planner-bird_train.json \
	--output_file data/llm_alignment/bird/llama-3b-validator-bird_train.jsonl \
	--model-name llama --mode train \
	 --n_return 6 --temperature 0.7 --api_host http://localhost:8003 --skip_planner --skip_fix

python jsonl2json.py --jsonl-file data/llm_alignment/bird/llama-3b-validator-bird_train.jsonl

rm data/llm_alignment/bird/llama-3b-fix-greedy-bird_train.jsonl
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/llm_alignment/bird/llama-3b-validator-bird_train.json \
	--output_file data/llm_alignment/bird/llama-3b-fix-greedy-bird_train.jsonl \
	--model-name llama --mode train \
	--n_return 1 --temperature 0.0 --api_host http://localhost:8003 --skip_planner --skip_validator

rm -r data/llm_alignment/bird/bird-p1-validator/
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file data/llm_alignment/bird/llama-3b-fix-greedy-bird_train.jsonl \
    --output_planner_file data/llm_alignment/bird/bird-p1-validator/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file data/llm_alignment/bird/bird-p1-validator/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file data/llm_alignment/bird/bird-p1-validator/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file data/llm_alignment/bird/bird-p1-validator/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file data/llm_alignment/bird/bird-p1-validator/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file data/llm_alignment/bird/bird-p1-validator/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl

# Use Feedback Editor Agent
rm processed_results.json
python modify_feedbacks.py --pred_file data/llm_alignment/bird/llama-3b-fix-greedy-bird_train.jsonl

rm -r data/llm_alignment/bird/bird-p1-validator-modify
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file data/llm_alignment/bird/llama-3b-fix-greedy-bird_train_progress.jsonl \
    --output_planner_file data/llm_alignment/bird/bird-p1-validator-modify/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file data/llm_alignment/bird/bird-p1-validator-modify/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file data/llm_alignment/bird/bird-p1-validator-modify/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file data/llm_alignment/bird/bird-p1-validator-modify/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file data/llm_alignment/bird/bird-p1-validator-modify/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file data/llm_alignment/bird/bird-p1-validator-modify/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl


# =================================================================================================
# GEN ORPO data for FIX
# =================================================================================================
CUDA_VISIBLE_DEVICES=1 vllm serve llama-3b-bird-fixed-fft-follow-validation/ --host 0.0.0.0 --port 8005 --dtype bfloat16 --max-model-len 4096 --disable-log-requests --served-model-name fixed --gpu-memory-utilization 0.8

rm data/llm_alignment/bird/llama-3b-validator-greedy-bird_train.jsonl
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/llm_alignment/bird/llama-3b-planner-bird_train.json \
	--output_file data/llm_alignment/bird/llama-3b-validator-greedy-bird_train.jsonl \
	--model-name llama --mode train \
	 --n_return 1 --temperature 0.0 --api_host http://localhost:8003 --skip_planner --skip_fix

python jsonl2json.py --jsonl-file data/llm_alignment/bird/llama-3b-validator-greedy-bird_train.jsonl

rm data/llm_alignment/bird/llama-3b-fix-sampling-bird_train.jsonl
PYTHONPATH=. python evaluate_end2end.py \
	--input_file data/llm_alignment/bird/llama-3b-validator-greedy-bird_train.json \
	--output_file data/llm_alignment/bird/llama-3b-fix-sampling-bird_train.jsonl \
	--model-name llama --mode train \
	--n_return 10 --temperature 0.7 --api_host http://localhost:8003 --skip_planner --skip_validator

rm -r data/llm_alignment/bird/bird-p1-fix/
PYTHONPATH=. python llm_alignment/build_rl_data.py \
    --input_file data/llm_alignment/bird/llama-3b-fix-sampling-bird_train.jsonl \
    --output_planner_file data/llm_alignment/bird/bird-p1-fix/dpo-llama-3-end2end-bird_train_planner.jsonl \
    --output_validator_select_file data/llm_alignment/bird/bird-p1-fix/dpo-llama-3-end2end-bird_train_validator_select.jsonl \
    --output_validator_condition_file data/llm_alignment/bird/bird-p1-fix/dpo-llama-3-end2end-bird_train_validator_condition.jsonl \
    --output_validator_join_file data/llm_alignment/bird/bird-p1-fix/dpo-llama-3-end2end-bird_train_validator_join.jsonl \
    --output_validator_order_file data/llm_alignment/bird/bird-p1-fix/dpo-llama-3-end2end-bird_train_validator_order.jsonl \
    --output_fixed_sql_file data/llm_alignment/bird/bird-p1-fix/dpo-llama-3-end2end-bird_train_fixed_sql.jsonl \
    --enable_advanced_fix_agent 
