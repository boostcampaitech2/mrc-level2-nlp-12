#!/bin/bash
python train.py --output_dir ./models/train_dataset --do_train True --do_eval True --overwrite_cache --overwrite_output_dir --model_name_or_path klue/roberta-base
# python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
