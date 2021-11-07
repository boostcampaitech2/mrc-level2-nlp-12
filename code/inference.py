import logging
import sys
import numpy as np
import torch

from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from typing import Callable, List, Dict, NoReturn, Tuple
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils.utils_qa import postprocess_qa_predictions, check_no_error
from utils.trainer_qa import QuestionAnsweringTrainer
from reader.conv import custom_model
from retrieval.sparse import es_dfr
from retrieval.dense import st, dpr
from retrieval.sparse import (
    TfidfRetriever,
    Bm25Retriever,
    EsBm25Retriever,
)


from arguments import (
    ModelArguments,
    DataTrainingArguments,
    DPRArguments

)


from arguments import ModelArguments, DataTrainingArguments, DPRArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DPRArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        dpr_args,
    ) = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )

    if model_args.model_type == "default":
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
        )

        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    elif model_args.model_type == "custom":
        model = (
            custom_model.CustomModelForQuestionAnswering()
        )  # conv-based custom model
        model.load_state_dict(
            torch.load("/opt/ml/code/models/roberta_conv_sum_st/pytorch_model.bin"),
            strict=False,
        )  # {your_path}
    else:
        raise ValueError("[ Model Type Not Found ] 해당하는 모델 유형이 없습니다.")

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args, dpr_args
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
    dpr_args: DPRArguments = None,
) -> DatasetDict:
    """
    wikipedia_documents.json 파일을 불러와서 retrieval을 실행하는 함수.

    Args:
        tokenize_fn (Callable[[str], List[str]]): 토크나이즈 함수, tokenizer가 아닌 tokenizer.tokenize를 인자로 받아야함.
        datasets (DatasetDict): DatasetDict 타입의 데이터셋
        training_args (TrainingArguments): 학습에 필요한 arguments
        data_args (DataTrainingArguments): 데이터에 관련된 arguments
        data_path (str, optional): 데이터셋의 경로. Defaults to "../data".
        context_path (str, optional): retrieval을 실행할 파일 경로. Defaults to "wikipedia_documents.json".
        dpr_args: (DPRArguments): DPR 관련된 arguments. Defaults to 'None'
        
    Returns:
        DatasetDict: retrieve 된 데이터셋을 리턴.
    """ 
    # Query에 맞는 Passage들을 Retrieval 합니다.
    # TFIDF | BM25 | ES_BM25 | ES_DFR | DPR | ST | HYBRID"
    if data_args.retriever_type == "TFIDF":
        retriever = TfidfRetriever(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_sparse_embedding()
    elif data_args.retriever_type == "BM25":
        retriever = Bm25Retriever(tokenize_fn=tokenize_fn)
    elif data_args.retriever_type == "ES_BM25":
        retriever = EsBm25Retriever()
    elif data_args.retriever_type == "ES_DFR":
        retriever = es_dfr.DFRRetriever()
        retriever._proc_init()
    elif data_args.retriever_type == "DPR":
        retriever = dpr.DensePassageRetrieval(dpr_args)
        retriever.load_passage_embedding()
    elif data_args.retriever_type == "ST":
        retriever = st.STRetriever()
        retriever._proc_init()
    elif data_args.retriever_type == "HYBRID":
        retriever = None

    if data_args.retriever_type == "TFIDF" and data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    """Reader 역할을 하는 MRC 모델을 실행하는 함수. 

    Args:
        data_args (DataTrainingArguments): 데이터 관련 arguments
        training_args (TrainingArguments): 훈련에 필요한 arguments
        model_args (ModelArguments): 모델 설정과 관련된 arguments
        datasets (DatasetDict): MRC에 사용되는 데이터셋
        tokenizer ([type]): 토크나이저
        model ([type]): 학습된 모델

    """ 
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        """MRC 모델에 맞게 데이터를 전처리하는 함수 

        Args:
            examples: 데이터셋

        Returns:
            토크나이즈된 데이터셋
        """
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        """MRC 모델에서 나온 output을 후처리해주는 함수

        Args:
            examples: 데이터셋
            features: column 종류
            predictions (Tuple[np.ndarray, np.ndarray]): 에측값을 튜플 형태로 받아옴
            training_args (TrainingArguments): 학습에 필요한 arguments

        Returns:
            EvalPrediction: do_predict 모드와 do_eval 모드에 맞는 Format의 output
        """
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
