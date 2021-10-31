import wandb
import os
import torch
import numpy as np
import prettyprinter as pp
import pprint
from typing import List, Callable, NoReturn, NewType, Any, Tuple, Optional

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizer,
    PreTrainedModel,
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import load_from_disk
from arguments import MaskedLanguageModelArguments, ModelArguments


class MLM_Dataset:
    def __init__(self, queries, tokenizer, model_name:str = 'klue/roberta-base', max_length:int = 32, cache_tokenization = False):
        self.queries = queries
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.cache_tokeniation = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokeniation:
            return self.tokenizer(
                self.queries[item],
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_token_type_ids=False if 'klue/roberta' in self.model_name else None,
                return_special_tokens_mask=True
            )
        
        if isinstance(self.queries[item], str):
            self.queries[item] = self.tokenizer(
                self.queries[item],
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_token_type_ids=False if 'klue/roberta' in self.model_name else None,
                return_special_tokens_mask=True
            )
        return self.queries[item]

    def __len__(self):
        return len(self.queries)


def main():
    parser = HfArgumentParser(
        MaskedLanguageModelArguments
    )

    mlm_args = parser.parse_args_into_dataclasses()[0]
    # output dir 수정하셔도 됩니다!
    mlm_args.output_dir = f'{mlm_args.output_dir}/' + mlm_args.model_name.replace('/', '_')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Load the model
    config_ = AutoConfig.from_pretrained(mlm_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(mlm_args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(mlm_args.model_name, config=config_)

    # wandb 설정
    # entity는 wandb login으로 자동 설정됩니다. entity를 변경하고 싶으시면 relogin하면 됩니다!
    os.environ["WANDB_ENTITY"] = "bc-ai-it-mrc" # 프로젝트 명
    os.environ["WANDB_PROJECT"] = "채워주세요" # 프로젝트 명 ex)TAPT-main

    ##### Load our training datasets
    dataset = load_from_disk(mlm_args.data_dir)
    MLM_train_dataset, MLM_dev_dataset = None, None

    if mlm_args.do_validation: # train / validation
        train_dataset = dataset['train'][:]['question']
        dev_dataset = dataset['validation'][:]['question']
        MLM_train_dataset = MLM_Dataset(
            queries=train_dataset,
            tokenizer=tokenizer,
            model_name=mlm_args.model_name,
            max_length=mlm_args.max_seq_length
        )
        MLM_dev_dataset = MLM_Dataset(
            queries=dev_dataset,
            tokenizer=tokenizer,
            model_name=mlm_args.model_name,
            max_length=mlm_args.max_seq_length
        )
    else: # only train(train+validation)
        train_dataset = dataset['train'][:]['question'] + dataset['validation'][:]['question']
        MLM_train_dataset = MLM_Dataset(
            queries=train_dataset,
            tokenizer=tokenizer,
            model_name=mlm_args.model_name,
            max_length=mlm_args.max_seq_length
        )

    # Random masking data collator based on the BERT paper.
    if mlm_args.do_whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=mlm_args.mlm_probability
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_args.mlm_probability
        )

    ##### Training arguments
    training_args = TrainingArguments(
        output_dir=mlm_args.output_dir,
        logging_dir=mlm_args.logging_dir,
        num_train_epochs=5,
        learning_rate=5e-05,
        evaluation_strategy="steps" if MLM_dev_dataset is not None else "no",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        #warmup_ratio=6e-02,
        warmup_steps=50,
        weight_decay=1e-02,
        load_best_model_at_end=True if MLM_dev_dataset is not None else False,
        save_total_limit=5,
        report_to='wandb',
        run_name=mlm_args.model_name
    )
    print(training_args)

    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=MLM_train_dataset,
        eval_dataset=MLM_dev_dataset,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # callbacks
        #optimizers=(optimizer, scheduler)  # optimizer and scheduler. fill with _ if you wanna use default setting.
    )

    print("TRAINING START...")
    trainer.train()

    print("TRAINING RESULT SAVING...")
    trainer.save_model(mlm_args.output_dir)
    tokenizer.save_pretrained(mlm_args.output_dir)

    print("TRAINING FINISHED!")


if __name__ == "__main__":
    main()

