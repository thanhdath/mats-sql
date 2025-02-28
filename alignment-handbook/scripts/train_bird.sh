export PYTHONPATH=src/
export CUDA_VISIBLE_DEVICES=0,1
# ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 1 scripts/run_orpo.py recipes/llama-3b-bird/orpo-planner.yaml --model_name_or_path=./output/llama-3b-bird-planner-fft --output_dir=output/reproduce/orpo-llama-3b-bird-planner

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_orpo.py recipes/llama-1b-bird/orpo-validator.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_orpo.py recipes/llama-1b-bird/orpo-fixed.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_orpo.py recipes/llama-1b-spider/orpo-validator.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_orpo.py recipes/llama-1b-spider/orpo-fixed.yaml