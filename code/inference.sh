#!/bin/bash
python inference.py \
	--output_dir ./outputs/koelectra\
    --do_predict \
    --overwrite_output_dir \
    --retriever_type "dense" \
	--model_name_or_path ./models/koelectra \
    --dataset_name /opt/ml/data/test_dataset \
	--tokenizer_name "monologg/koelectra-base-v3-finetuned-korquad" \
	# --tokenizer_name klue/roberta-large \
	# --model_name_or_path "monologg/koelectra-base-v3-finetuned-korquad" \
	# --do_predict 1 \
    # --retriever_type "dense" \
    # --overwrite_output_dir 1 \