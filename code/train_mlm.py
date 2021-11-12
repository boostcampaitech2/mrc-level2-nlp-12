import wandb
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    set_seed,
)
from datasets import load_from_disk
from arguments import MaskedLanguageModelArguments, ModelArguments


class MLM_Dataset:
    def __init__(
        self,
        queries,
        tokenizer,
        model_name:str = 'klue/roberta-base',
        max_length:int = 32,
        cache_tokenization:bool = False
    ):
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
        (MaskedLanguageModelArguments, TrainingArguments)
    )
    mlm_args, training_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # load model
    config_ = AutoConfig.from_pretrained(mlm_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(mlm_args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(mlm_args.model_name, config=config_)

    # wandb setting
    # wnadb entity is automatically set up after wandb login. If you want to change it, then relogin.
    os.environ["WANDB_ENTITY"] = "_" # WANDB entity name e.g. bc-ai-it-mrc
    os.environ["WANDB_PROJECT"] = "_" # WANDB project name e.g. TAPT-main

    ##### load datasets
    dataset = load_from_disk(mlm_args.data_dir)
    MLM_train_dataset, MLM_dev_dataset = None, None

    if mlm_args.do_validation:
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
    else:
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
    training_args.output_dir=f"{mlm_args.mlm_dir}/" + mlm_args.model_name.replace('/', '_')
    training_args.logging_dir=f"{mlm_args.mlm_dir}/logs"
    training_args.num_train_epochs=5
    training_args.learning_rate=5e-05
    training_args.evaluation_strategy="steps" if MLM_dev_dataset is not None else "no"
    training_args.per_device_train_batch_size=64
    training_args.per_device_eval_batch_size=64
    training_args.eval_steps=50
    training_args.save_steps=50
    training_args.logging_steps=50
    training_args.warmup_steps=50
    training_args.weight_decay=1e-02
    training_args.load_best_model_at_end=True if MLM_dev_dataset is not None else False
    training_args.save_total_limit=5
    training_args.report_to='wandb'

    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        logging_dir=training_args.logging_dir,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        evaluation_strategy=training_args.evaluation_strategy,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
        logging_steps=training_args.logging_steps,
        warmup_steps=training_args.warmup_steps,
        weight_decay=training_args.weight_decay,
        load_best_model_at_end=training_args.load_best_model_at_end,
        save_total_limit=training_args.save_total_limit,
        report_to=training_args.report_to,
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
    )

    print("TRAINING START...")
    trainer.train()

    print("TRAINING RESULT SAVING...")
    trainer.save_model(mlm_args.output_dir)
    tokenizer.save_pretrained(mlm_args.output_dir)

    print("TRAINING FINISHED!")

if __name__ == "__main__":
    main()

