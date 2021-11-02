python dpr_trainer.py \
        --project_name "T2050-retrieval-dev" \
        --entity_name "bc-ai-it-mrc" \
        --retriever_run_name "preprocessed" \
        --output_dir "./output_dir" \
        --eval_steps 300 \
        --num_neg 12 \
        --train_batch_size 2 \
