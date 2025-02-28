# n=9
# temperature=1.0
# seed=100
# mkdir logs/
# mkdir logs/dr-spider
# for test_set in DB_schema_synonym DB_DBcontent_equivalence DB_schema_abbreviation NLQ_column_value SQL_DB_text SQL_DB_number SQL_comparison NLQ_column_carrier NLQ_multitype NLQ_others NLQ_keyword_synonym SQL_NonDB_number  NLQ_column_synonym  NLQ_keyword_carrier NLQ_value_synonym NLQ_column_attribute SQL_sort_order
# do
# 	eval_file=data/evaluate/orpo-llama-3-iter-3-planner-dr_spider_${test_set}.jsonl
# 	# rm $eval_file
# 	PYTHONPATH=. python evaluate_end2end.py \
# 		--input_file  data/schema_insight_dr_spider_text2sql_${test_set}.json  \
# 		--output_file $eval_file \
# 		--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --only_planner --seed $seed --n_processes 32
	
# 	rm progress_dr.pkl
# 	python -u check_correct_recall.py --pred_file $eval_file --progress_file progress_dr.pkl > logs/dr-spider/log-orpo-planner-dr_spider-$test_set-n$n-temp$temperature-seed$seed.txt
# done

n=9
temperature=1.0
seed=100
mkdir logs/
mkdir logs/dr-spider
for test_set in SQL_comparison NLQ_column_carrier NLQ_multitype NLQ_others NLQ_keyword_synonym SQL_NonDB_number  NLQ_column_synonym  NLQ_keyword_carrier NLQ_value_synonym NLQ_column_attribute SQL_sort_order
do
	python jsonl2json.py --jsonl-file data/evaluate/orpo-llama-3-iter-3-planner-dr_spider_${test_set}.jsonl

	eval_file=data/evaluate/orpo-llama-3-iter-3-end2end-dr_spider_${test_set}.jsonl
	# rm $eval_file
	PYTHONPATH=. python evaluate_end2end.py \
		--input_file data/evaluate/orpo-llama-3-iter-3-planner-dr_spider_${test_set}.json  \
		--output_file $eval_file \
		--model-name llama --mode test --n_return $n --temperature $temperature --api_host http://192.168.1.118:8003 --skip_planner --seed $seed --n_processes 32 --skip_validator_join --skip_selection

	rm progress_dr.pkl
	python -u check_correct_recall.py --pred_file $eval_file --progress_file progress_dr.pkl > logs/dr-spider/log-orpo-llama-3-iter-3-end2end-dr_spider-$test_set-n$n-temp$temperature-seed$seed.txt
	mv results_spider.pkl logs/results-dr-spider-end2end-$test_set.pkl
done

