# # Evaluate 
mkdir logs
mkdir logs/spider-dev
for n in 9
do
	temperature=1.0
	eval_file=data/evaluate/orpo-planner-spider_dev-greedy-and-sampling-seed$seed.jsonl
	# rm $eval_file
	PYTHONPATH=. python evaluate_end2end.py \
		--input_file  data/schema_insight_spider_dev_text2sql.json  \
		--output_file $eval_file \
		--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --only_planner --seed $seed --n_processes 32
	
	rm progress.pkl
	python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-dev/log-orpo-planner-spider_dev-greedy-and-sampling-seed$seed-n$n.txt
	mv results-spider.pkl logs/results-spider-dev-greedy-and-sampling.pkl
done

# # spider_realistic
# mkdir logs
# mkdir logs/spider-realistic
# for n in 9
# do
# 	temperature=1.0
# 	for seed in 100
# 	do 
# 		eval_file=data/evaluate/orpo-planner-spider_realistic-greedy-and-sampling-seed$seed.jsonl
# 		# rm $eval_file
# 		PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file  data/schema_insight_spider_realistic_text2sql.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --only_planner --seed $seed --n_processes 32
		
# 		rm progress.pkl
# 		python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-realistic/log-orpo-planner-spider_realistic-greedy-and-sampling-seed$seed-n$n.txt
# 	done
# done

# mkdir logs
# mkdir logs/spider-dk
# for n in 9
# do
# 	temperature=1.0
# 	for seed in 100
# 	do 
# 		eval_file=data/evaluate/orpo-planner-spider_dk-greedy-and-sampling-seed$seed.jsonl
# 		# rm $eval_file
# 		PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file  data/schema_insight_spider_dk_text2sql.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --only_planner --seed $seed --n_processes 32
		
# 		rm progress.pkl
# 		python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-dk/log-orpo-planner-spider_dk-greedy-and-sampling-seed$seed-n$n.txt
# 	done
# done

# mkdir logs
# mkdir logs/spider-syn
# for n in 9
# do
# 	temperature=1.0
# 	for seed in 100
# 	do 
# 		eval_file=data/evaluate/orpo-planner-spider_syn-greedy-and-sampling-seed$seed.jsonl
# 		# rm $eval_file
# 		PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file  data/schema_insight_spider_syn_text2sql.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --only_planner --seed $seed --n_processes 32
		
# 		rm progress.pkl
# 		python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-syn/log-orpo-planner-spider_syn-greedy-and-sampling-seed$seed-n$n.txt
# 		mv results-spider.pkl logs/results-spider-syn-greedy-and-sampling.pkl
# 	done
# done


# Generate Validation and Fix
# python jsonl2json.py --jsonl-file data/evaluate/orpo-planner-spider_syn-greedy-and-sampling-seed100.jsonl
# python jsonl2json.py --jsonl-file data/evaluate/orpo-planner-spider_dk-greedy-and-sampling-seed100.jsonl
# python jsonl2json.py --jsonl-file data/evaluate/orpo-planner-spider_realistic-greedy-and-sampling-seed100.jsonl
# python jsonl2json.py --jsonl-file data/evaluate/orpo-planner-spider_dev-greedy-and-sampling-seed100.jsonl

# n=9
# temperature=0.0
# seed=100

# eval_file=data/evaluate/orpo-end2end-spider_syn-greedy-and-sampling-seed$seed.jsonl
# rm $eval_file
# PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file data/evaluate/orpo-planner-spider_syn-greedy-and-sampling-seed100.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator_join
# rm progress.pkl
# python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-syn/log-orpo-end2end-spider_syn-greedy-and-sampling-seed$seed-n$n.txt
# mv results-spider.pkl logs/results-spider-syn-end2end.pkl

# eval_file=data/evaluate/orpo-end2end-spider_dk-greedy-and-sampling-seed$seed.jsonl
# rm $eval_file
# PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file data/evaluate/orpo-planner-spider_dk-greedy-and-sampling-seed100.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator_join
# rm progress.pkl
# python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-dk/log-orpo-end2end-spider_dk-greedy-and-sampling-seed$seed-n$n.txt
# mv results-spider.pkl logs/results-spider-dk-end2end.pkl

# eval_file=data/evaluate/orpo-end2end-spider_realistic-greedy-and-sampling-seed$seed.jsonl
# rm $eval_file
# PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file data/evaluate/orpo-planner-spider_realistic-greedy-and-sampling-seed100.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator_join
# rm progress.pkl
# python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-realistic/log-orpo-end2end-spider_realistic-greedy-and-sampling-seed$seed-n$n.txt
# mv results-spider.pkl logs/results-spider-realistic-end2end.pkl

# eval_file=data/evaluate/orpo-end2end-spider_dev-greedy-and-sampling-seed$seed.jsonl
# rm $eval_file
# PYTHONPATH=. python evaluate_end2end.py \
# 			--input_file data/evaluate/orpo-planner-spider_dev-greedy-and-sampling-seed100.json  \
# 			--output_file $eval_file \
# 			--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator_join
# rm progress.pkl
# python -u check_correct_recall.py --pred_file $eval_file --progress_file progress.pkl > logs/spider-dev/log-orpo-end2end-spider_dev-greedy-and-sampling-seed$seed-n$n.txt
# mv results-spider.pkl logs/results-spider-dev-end2end.pkl


# Evaluate Selection
python jsonl2json.py --jsonl-file data/evaluate/orpo-end2end-spider_syn-greedy-and-sampling-seed100.jsonl
python jsonl2json.py --jsonl-file data/evaluate/orpo-end2end-spider_dk-greedy-and-sampling-seed100.jsonl
python jsonl2json.py --jsonl-file data/evaluate/orpo-end2end-spider_realistic-greedy-and-sampling-seed100.jsonl
python jsonl2json.py --jsonl-file data/evaluate/orpo-end2end-spider_dev-greedy-and-sampling-seed100.jsonl

n=9
temperature=0.0
seed=100
eval_file=data/evaluate/orpo-selection-spider_syn-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end_update.py \
	--input_file data/evaluate/orpo-end2end-spider_syn-greedy-and-sampling-seed100.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator --skip_fix
python compute_acc.py --pred_file $eval_file > logs/spider-syn/log-orpo-selection-spider_syn-greedy-and-sampling-seed$seed-n$n.txt


eval_file=data/evaluate/orpo-selection-spider_dk-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end_update.py \
	--input_file data/evaluate/orpo-end2end-spider_dk-greedy-and-sampling-seed100.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator --skip_fix
python compute_acc.py --pred_file $eval_file > logs/spider-dk/log-orpo-selection-spider_dk-greedy-and-sampling-seed$seed-n$n.txt

eval_file=data/evaluate/orpo-selection-spider_realistic-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end_update.py \
	--input_file data/evaluate/orpo-end2end-spider_realistic-greedy-and-sampling-seed100.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator --skip_fix
python compute_acc.py --pred_file $eval_file > logs/spider-realistic/log-orpo-selection-spider_realistic-greedy-and-sampling-seed$seed-n$n.txt

eval_file=data/evaluate/orpo-selection-spider_dev-greedy-and-sampling-seed$seed.jsonl
rm $eval_file
PYTHONPATH=. python evaluate_end2end_update.py \
	--input_file data/evaluate/orpo-end2end-spider_dev-greedy-and-sampling-seed100.json  \
	--output_file $eval_file \
	--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://localhost:8003 --skip_planner --seed 100 --n_processes 32 --skip_validator --skip_fix
python compute_acc.py --pred_file $eval_file > logs/spider-dev/log-orpo-selection-spider_dev-greedy-and-sampling-seed$seed-n$n.txt

