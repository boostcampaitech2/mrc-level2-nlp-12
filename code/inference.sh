#!/bin/bash
python inference.py \
	--output_dir ./outputs/dpr-optim_mlm_optim\
    --do_predict \
    --overwrite_output_dir \
    --retriever_type "dense" \
	--model_name_or_path ./models/klue-mlm-optim \
    --dataset_name /opt/ml/data/test_dataset \
	--tokenizer_name "klue/roberta-large" \
	# --tokenizer_name klue/roberta-large \
	# --model_name_or_path "monologg/koelectra-base-v3-finetuned-korquad" \
	# --do_predict 1 \
    # --retriever_type "dense" \
    # --overwrite_output_dir 1 \