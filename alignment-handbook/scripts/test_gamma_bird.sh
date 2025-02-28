export PYTHONPATH=src/
export CUDA_VISIBLE_DEVICES=0,1

for beta in 0.0 0.25 0.5 1.0 0.75
do
    ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29504 --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes 2 scripts/run_orpo.py recipes/llama-3b-bird/orpo-planner.yaml --model_name_or_path=./output/reproduce/llama-3b-bird-planner-fft-no-filter  --output_dir=output/param-sensitivity/orpo-llama-3b-bird-planner-beta-$beta --save_strategy=no
done
