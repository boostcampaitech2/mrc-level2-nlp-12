from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    best_model: str = field(
        default="./models/best_model",
        metadata={
            "help": "Path to best model"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=1,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )


@dataclass
class MaskedLanguageModelArguments:
    """
    Arguments pertaining to MLM.
    """

    model_name: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "model identifier from huggingface.co/models"
        },
    )
    do_whole_word_mask: bool = field(
        default=False,
        metadata={
            "help": "If set to true, whole words are masked."
        },
    )
    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "Probability that a word is replaced by a [MASK] token."
        },
    )
    do_validation: bool = field(
        default=True,
        metadata={
            "help": "If set to False, train/validation data will be merged into train data."
        },
    )
    max_seq_length: int = field(
        default=40,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    data_dir: Optional[str] = field(
        default="../data/train_dataset",
        metadata={
            "help": "training data directory name."
        },
    )
    logging_dir: Optional[str] = field(
        default="./models/pretrained_mlm/logs",
        metadata={
            "help": "training log directory name."
        },
    )
    output_dir: Optional[str] = field(
        default="./models/pretrained_mlm",
        metadata={
            "help": "training output directory name."
        },
    )


@dataclass
class EnsembleArguments:
    """
    Arguments pertaining to ensemble.
    """

    nbest_dir: str = field(
        default="./ensemble/nbests",
        metadata={
            "help": "Prediction output directory. (default: ./ensemble/nbests)"
        }
    )
    output_dir: str = field(
        default="./ensemble/predictions",
        metadata={
            "help": "Prediction output directory. (default: ./ensemble/predictions)"
        }
    )
    do_hard_voting: bool = field(
        default=True,
        metadata={
            "help": "Activate hard voting. Set False if you do not want to do hard voting."
        }
    )
    do_soft_voting: bool = field(
        default=True,
        metadata={
            "help": "Activate soft voting. Set False if you do not want to do soft voting."
        }
    )