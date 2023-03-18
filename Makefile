# RM
dataset:
	cd neg_udl/ && python data/make_dataset.py
wiki:
	cd neg_udl/ && python data/make_wiki.py
owt:
	cd neg_udl/ && python data/make_owt.py
cc_news:
	cd neg_udl/ && python data/make_cc_news.py
bc:
	cd neg_udl/ && python data/make_bookcorpus.py
evaluate:
	python evaluation/evaluation.py

exp_1_filtered:
	python main.py --config-name exp1_config
exp_1_filtered_DEBUG:
	python main.py --config-name exp1_config 'data.path=./data/processed/roberta_filtered/debug.txt'

exp_2_mlm+sup:
	python main.py --config-name exp2_config

exp_2_mlm+sup_DEBUG:
	python main.py --config-name exp2_config 'data.path=./data/processed/mixed_experiment/debug.txt'

exp_3_mlm:
	python main.py --config-name exp3_config

exp_3_mlm-DEBUG:
	python main.py --config-name exp3_config 'data.path=./data/processed/mlm_mixed/debug.txt'

exp_3+_mlm:
	python main.py --config-name