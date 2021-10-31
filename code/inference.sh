#!/bin/bash
python inference.py \
	--output_dir ./outputs/dpr_klue-roberta-large-mlm \
    --do_predict \
    --overwrite_output_dir \
    --retriever_type "dense" \
	--model_name_or_path ./models/klue-mlm \
    --dataset_name /opt/ml/data/test_dataset \
	--tokenizer_name klue/roberta-large \
	# --do_predict 1 \
    # --retriever_type "dense" \
    # --overwrite_output_dir 1 \