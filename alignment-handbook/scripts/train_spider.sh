export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src/
#accelerate launch --main_process_port 29502 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_sft.py recipes/llama-3b-spider/planner-fft.yaml
accelerate launch --main_process_port 29502 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 1 scripts/run_orpo.py recipes/llama-3b-spider/orpo-planner-iter-3.yaml
# accelerate launch --main_process_port 29502 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_sft.py recipes/llama-3b-spider/sql-gt-fft.yaml


