python train.py \
        --model_name_or_path klue/roberta-large \
        --output_dir ./models/klue/roberta-large \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --logging_steps 100 \
        --eval_step 250 \
        --overwrite_cache \
        --overwrite_output_dir \
        --preprocessing_num_workers 4 \
        --run_name "[김재현]-klue/roberta-large" \
        --learning_rate 1e-5 \
        --warmup_ratio 0.2 \
        # --max_seq_length 256 \
        # --doc_stride 128 \
        # --max_answer_length 64 \
# python train.py \
#         --model_name_or_path /opt/ml/mrc-level2-nlp-12/code/models/best_model \
#         --output_dir ./models/klue-roberta-large-vanilla\
#         --do_eval \
#         --do_train \
#         --evaluation_strategy steps \
#         --logging_steps 100 \
#         --eval_step 250 \
#         --overwrite_cache \
#         --overwrite_output_dir \
#         --preprocessing_num_workers 4 \
#         --run_name "[김재현]-klue/roberta-large-vanilla-saved" \
        # --model_name_or_path klue/roberta-large \
        # --warmup_steps 400 \
        # --model_name_or_path klue/roberta-large \

# optional arguments:
#   -h, --help            show this help message and exit
#   --model_name_or_path MODEL_NAME_OR_PATH
#                         Path to pretrained model or model identifier from huggingface.co/models
#   --config_name CONFIG_NAME
#                         Pretrained config name or path if not the same as model_name
#   --tokenizer_name TOKENIZER_NAME
#                         Pretrained tokenizer name or path if not the same as model_name
#   --dataset_name DATASET_NAME
#                         The name of the dataset to use.
#   --overwrite_cache [OVERWRITE_CACHE]
#                         Overwrite the cached training and evaluation sets
#   --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
#                         The number of processes to use for the preprocessing.
#   --max_seq_length MAX_SEQ_LENGTH
#                         The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
#   --pad_to_max_length [PAD_TO_MAX_LENGTH]
#                         Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch (which can
#                         be faster on GPU but will be slower on TPU).
#   --doc_stride DOC_STRIDE
#                         When splitting up a long document into chunks, how much stride to take between chunks.
#   --max_answer_length MAX_ANSWER_LENGTH
#                         The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.
#   --no_eval_retrieval   Whether to run passage retrieval using sparse embedding.
#   --eval_retrieval [EVAL_RETRIEVAL]
#                         Whether to run passage retrieval using sparse embedding.
#   --num_clusters NUM_CLUSTERS
#                         Define how many clusters to use for faiss.
#   --top_k_retrieval TOP_K_RETRIEVAL
#                         Define how many top-k passages to retrieve based on similarity.
#   --use_faiss [USE_FAISS]
#                         Whether to build with faiss
#   --output_dir OUTPUT_DIR
#                         The output directory where the model predictions and checkpoints will be written.
#   --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
#                         Overwrite the content of the output directory.Use this to continue training if output_dir points to a checkpoint directory.
#   --do_train [DO_TRAIN]
#                         Whether to run training.
#   --do_eval [DO_EVAL]   Whether to run eval on the dev set.
#   --do_predict [DO_PREDICT]
#                         Whether to run predictions on the test set.
#   --evaluation_strategy {no,steps,epoch}
#                         The evaluation strategy to use.
#   --prediction_loss_only [PREDICTION_LOSS_ONLY]
#                         When performing evaluation and predictions, only returns the loss.
#   --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
#                         Batch size per GPU/TPU core/CPU for training.
#   --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
#                         Batch size per GPU/TPU core/CPU for evaluation.
#   --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
#                         Deprecated, the use of `--per_device_train_batch_size` is preferred. Batch size per GPU/TPU core/CPU for training.
#   --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
#                         Deprecated, the use of `--per_device_eval_batch_size` is preferred.Batch size per GPU/TPU core/CPU for evaluation.
#   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
#                         Number of updates steps to accumulate before performing a backward/update pass.
#   --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
#                         Number of predictions steps to accumulate before moving the tensors to the CPU.
#   --learning_rate LEARNING_RATE
#                         The initial learning rate for AdamW.
#   --weight_decay WEIGHT_DECAY
#                         Weight decay for AdamW if we apply some.
#   --adam_beta1 ADAM_BETA1
#                         Beta1 for AdamW optimizer
#   --adam_beta2 ADAM_BETA2
#                         Beta2 for AdamW optimizer
#   --adam_epsilon ADAM_EPSILON
#                         Epsilon for AdamW optimizer.
#   --max_grad_norm MAX_GRAD_NORM
#                         Max gradient norm.
#   --num_train_epochs NUM_TRAIN_EPOCHS
#                         Total number of training epochs to perform.
#   --max_steps MAX_STEPS
#                         If > 0: set total number of training steps to perform. Override num_train_epochs.
#   --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
#                         The scheduler type to use.
#   --warmup_ratio WARMUP_RATIO
#                         Linear warmup over warmup_ratio fraction of total steps.
#   --warmup_steps WARMUP_STEPS
#                         Linear warmup over warmup_steps.
#   --logging_dir LOGGING_DIR
#                         Tensorboard log dir.
#   --logging_strategy {no,steps,epoch}
#                         The logging strategy to use.
#   --logging_first_step [LOGGING_FIRST_STEP]
#                         Log the first global_step
#   --logging_steps LOGGING_STEPS
#                         Log every X updates steps.
#   --save_strategy {no,steps,epoch}
#                         The checkpoint save strategy to use.
#   --save_steps SAVE_STEPS
#                         Save checkpoint every X updates steps.
#   --save_total_limit SAVE_TOTAL_LIMIT
#                         Limit the total amount of checkpoints.Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints
#   --no_cuda [NO_CUDA]   Do not use CUDA even when it is available
#   --seed SEED           Random seed that will be set at the beginning of training.
#   --fp16 [FP16]         Whether to use 16-bit (mixed) precision instead of 32-bit
#   --fp16_opt_level FP16_OPT_LEVEL
#                         For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html
#   --fp16_backend {auto,amp,apex}
#                         The backend to be used for mixed precision.
#   --fp16_full_eval [FP16_FULL_EVAL]
#                         Whether to use full 16-bit precision evaluation instead of 32-bit
#   --local_rank LOCAL_RANK
#                         For distributed training: local_rank
#   --tpu_num_cores TPU_NUM_CORES
#                         TPU: Number of TPU cores (automatically passed by launcher script)
#   --tpu_metrics_debug [TPU_METRICS_DEBUG]
#                         Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics
#   --debug [DEBUG]       Whether to print debug metrics on TPU
#   --dataloader_drop_last [DATALOADER_DROP_LAST]
#                         Drop the last incomplete batch if it is not divisible by the batch size.
#   --eval_steps EVAL_STEPS
#                         Run an evaluation every X steps.
#   --dataloader_num_workers DATALOADER_NUM_WORKERS
#                         Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.
#   --past_index PAST_INDEX
#                         If >=0, uses the corresponding part of the output as the past state for next step.
#   --run_name RUN_NAME   An optional descriptor for the run. Notably used for wandb logging.
#   --disable_tqdm DISABLE_TQDM
#                         Whether or not to disable the tqdm progress bars.
#   --no_remove_unused_columns
#                         Remove columns not required by the model when using an nlp.Dataset.
#   --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
#                         Remove columns not required by the model when using an nlp.Dataset.
#   --label_names LABEL_NAMES [LABEL_NAMES ...]
#                         The list of keys in your dictionary of inputs that correspond to the labels.
#   --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
#                         Whether or not to load the best model found during training at the end of training.
#   --metric_for_best_model METRIC_FOR_BEST_MODEL
#                         The metric to use to compare two different models.
#   --greater_is_better GREATER_IS_BETTER
#                         Whether the `metric_for_best_model` should be maximized or not.
#   --ignore_data_skip [IGNORE_DATA_SKIP]
#                         When resuming training, whether or not to skip the first epochs and batches to get to the same training data.
#   --sharded_ddp SHARDED_DDP
#                         Whether or not to use sharded DDP training (in distributed training only). The base option should be `simple`, `zero_dp_2` or `zero_dp_3` and you can
#                         add CPU-offload to `zero_dp_2` or `zero_dp_3` like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or with the
#                         same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.
#   --deepspeed DEEPSPEED
#                         Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict
#   --label_smoothing_factor LABEL_SMOOTHING_FACTOR
#                         The label smoothing epsilon to apply (zero means no label smoothing).
#   --adafactor [ADAFACTOR]
#                         Whether or not to replace AdamW by Adafactor.
#   --group_by_length [GROUP_BY_LENGTH]
#                         Whether or not to group samples of roughly the same length together when batching.
#   --length_column_name LENGTH_COLUMN_NAME
#                         Column name with precomputed lengths to use when grouping by length.
#   --report_to REPORT_TO [REPORT_TO ...]
#                         The list of integrations to report the results and logs to.
#   --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
#                         When using distributed training, the value of the flag `find_unused_parameters` passed to `DistributedDataParallel`.
#   --no_dataloader_pin_memory
#                         Whether or not to pin memory for DataLoader.
#   --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
#                         Whether or not to pin memory for DataLoader.
#   --skip_memory_metrics [SKIP_MEMORY_METRICS]
#                         Whether or not to skip adding of memory profiler reports to metrics.
#   --mp_parameters MP_PARAMETERS
#                         Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer