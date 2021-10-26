from dpr_train_utils import DPRTrainer
from dpr_model import DPRetrieval
import os.path as path
import json
from datasets import load_from_disk


def load_wiki(wiki_path=None):
    if wiki_path == None:
        with open(path.join("/opt/ml/", "data", "wikipedia_documents.json"), "r") as f:
            wiki = json.load(f)
    else:
        with open(path.join(wiki_path, "wikipedia_documents.json"), "r") as f:
            wiki = json.load(f)
    return wiki


def load_datasets(data_path="/opt/ml/data/train_dataset"):
    dataset = load_from_disk(data_path)
    return dataset


if __name__ == "__main__":
    from arguments import ModelArguments, DataTrainingArguments, RetrievalArguments
    from transformers import (
        HfArgumentParser,
        TrainingArguments,
        set_seed,
        AutoTokenizer,
    )

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, RetrievalArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        retrieval_args,
    ) = parser.parse_args_into_dataclasses()

    set_seed(42)
    TOKENIZER_NAME = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    print(f"tokenizer : {TOKENIZER_NAME}")

    wiki_data = load_wiki("/opt/ml/data")
    dataset = load_datasets()

    retriever = DPRetrieval(
        retrieval_args, tokenizer, wiki_data, dataset["train"], dataset["validation"]
    )
    q_encoder, p_encoder = retriever._load_encoder()
    # train_dataset = retriever._load_dataset()
    # eval_dataset = retriever._load_eval_dataset()
    trainer = DPRTrainer(
        retrieval_args, tokenizer, wiki_data, dataset["train"], dataset["validation"]
    )

    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=retrieval_args.lr,
        per_device_train_batch_size=retrieval_args.train_batch_size,
        per_device_eval_batch_size=retrieval_args.eval_batch_size,
        num_train_epochs=retrieval_args.epochs,
        weight_decay=0.01,
        gradient_accumulation_steps=4,  # 메모리 효율
        logging_steps=100,
        eval_steps=100,
    )

    trainer.train(
        args,
        p_encoder,
        q_encoder,
        project="T2050-retrieval-dev",
        entity="bc-ai-it-mrc",
        runname="metric_text",
    )

