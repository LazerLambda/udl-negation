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