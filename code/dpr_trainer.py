from retrieval.dense.dpr import DPRTrainer, DPRDataset

import os.path as path


def main(
    model_args, data_args, training_args, retrieval_args,
):
    dpr_dataset = DPRDataset(
        path.join(retrieval_args.train_data_dir, "wikipedia_documents.json"),
        path.join(retrieval_args.train_data_dir, retrieval_args.train_data_name),
    )

    TOKENIZER_NAME = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    print(f"tokenizer : {TOKENIZER_NAME}")

    wiki_data = dpr_dataset.load_wiki_data()
    train_data = dpr_dataset.load_train_data()
    valid_data = dpr_dataset.load_valid_data()

    trainer = DPRTrainer(retrieval_args, tokenizer, wiki_data, train_data, valid_data)

    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="steps",
        learning_rate=retrieval_args.lr,
        per_device_train_batch_size=retrieval_args.train_batch_size,
        per_device_eval_batch_size=retrieval_args.eval_batch_size,
        num_train_epochs=retrieval_args.epochs,
        weight_decay=retrieval_args.dpr_weight_decay,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,  # 메모리 효율
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
    )

    trainer.train(
        args,
        project=retrieval_args.project_name,
        entity=retrieval_args.entity_name,
        runname=retrieval_args.retriever_run_name,
    )


if __name__ == "__main__":
    from arguments import ModelArguments, DataTrainingArguments, DPRArguments
    from transformers import (
        HfArgumentParser,
        TrainingArguments,
        set_seed,
        AutoTokenizer,
    )

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DPRArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        retrieval_args,
    ) = parser.parse_args_into_dataclasses()

    set_seed(42)

    main(model_args, data_args, training_args, retrieval_args)

