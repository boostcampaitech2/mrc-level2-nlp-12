#!/bin/bash
python inference.py\
	--output_dir ./outputs/test_dataset/\
       	--dataset_name ../data/test_dataset/\
	--model_name_or_path ./models/train_dataset/checkpoint-2750/\
	--do_predict \
	--tokenizer_name klue/roberta-base\
