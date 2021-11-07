from transformers import (
    AutoModel,
    AutoConfig,
)
import torch.nn as nn
from torch.utils.data import TensorDataset
import os.path as path
import numpy as np
from .dpr_dataset import BM25Data
from contextlib import contextmanager
import time
import tqdm
import pickle


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class Encoder(nn.Module):
    """The encoder to use for embedding questions and passages.

    Args:
        model_checkpoint: huggingface pretrained model
    """

    def __init__(self, model_checkpoint):
        super(Encoder, self).__init__()
        self.model_checkpoint = model_checkpoint
        config = AutoConfig.from_pretrained(self.model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        return pooled_output

    # DPRetrieval arguments list
    """
    q_encoder_path
    p_encoder_path
    model_checkpoint
    neg_strategy
    num_neg
    """


class DPRetrieval:
    """A common class for all retrieval methods
    """

    def __init__(self, args, tokenizer, wiki_data, train_data, eval_data):
        # set_seed(42)
        self.args = args
        self.tokenizer = tokenizer
        self.wiki = wiki_data
        self.train_dataset = train_data
        self.eval_dataset = eval_data
        self.contexts = [v["text"] for v in self.wiki.values()]
        self.context_ids = [v["document_id"] for v in self.wiki.values()]
        self.bm25 = BM25Data(tokenizer.tokenize, wiki_data)

    def _load_encoder(self):
        """Load encoder from model checkpoint
        
        Returns:
            q_encoder, p_encoder: pair of q & p encoder
        """
        print("--- Load Encoders from model checkpoint ---")
        q_encoder = Encoder(self.args.model_checkpoint)
        p_encoder = Encoder(self.args.model_checkpoint)

        return q_encoder, p_encoder

    def _load_dataset(self):
        """Set in-batch negative sampling train datasets

        Returns:
            final_train_dataset (TensorDataset): in-batch negative sampling datasets
        """
        if path.isfile("final_train_dataset.bin"):
            print("---- load saved dataset ----")
            with open("final_train_dataset.bin", "rb") as file:
                final_train_dataset = pickle.load(file)
            return final_train_dataset
        else:
            print("---- set dataset ----")
            corpus = np.array(self.train_dataset["context"])
            cor_titles = np.array(self.train_dataset["title"])
            p_with_neg = []
            titles = []
            if self.args.neg_strategy == "random":
                for t, c in tqdm.tqdm(
                    zip(self.train_dataset["title"], self.train_dataset["context"]),
                    desc="setting in-batch dataset (option: random sampling)",
                ):
                    titles.append(t)
                    while True:
                        neg_idxs = np.random.randint(
                            len(corpus), size=self.args.num_neg
                        )
                        if not c in corpus[neg_idxs]:
                            p_neg = corpus[neg_idxs]
                            p_neg_title = cor_titles[neg_idxs]
                            p_with_neg.append(c)
                            p_with_neg.extend(p_neg)
                            titles.extend(p_neg_title)
                            break

            elif self.args.neg_strategy == "BM_Gold":
                with timer("BM Gold neg sampling"):
                    self.bm25.get_sparse_embedding()
                    for t, c, q in tqdm.tqdm(
                        zip(
                            self.train_dataset["title"],
                            self.train_dataset["context"],
                            self.train_dataset["question"],
                        ),
                        desc="setting in-batch dataset (option: BM Gold neg sampling)",
                    ):
                        titles.append(t)
                        while True:
                            neg_idxs = np.random.randint(
                                len(corpus), size=self.args.num_neg - 1
                            )
                            if not c in corpus[neg_idxs]:
                                p_neg = corpus[neg_idxs]
                                p_with_neg.append(c)
                                p_neg_title = cor_titles[neg_idxs]
                                # BM25 top1 context 넣기

                                bm_score, bm_context, bm_indice = self.bm25.retrieve(
                                    q, 50
                                )
                                i = 0
                                while True:
                                    assert i < 50, f"bm25의 top50 까지 {c}와 같아 에러 발생"
                                    if bm_context[i] != c:
                                        p_with_neg.append(bm_context[i])
                                        titles.append(
                                            self.wiki[str(bm_indice[i])]["title"]
                                        )
                                        break
                                    i += 1
                                p_with_neg.extend(p_neg)
                                titles.extend(p_neg_title)
                                break

            q_seqs = self.tokenizer(
                self.train_dataset["question"],
                padding="max_length",
                truncation=True,
                max_length=100,
                return_tensors="pt",
                return_token_type_ids=(
                    False if "roberta" in self.args.model_checkpoint else True
                ),  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            )
            p_seqs = self.tokenizer(
                titles,
                p_with_neg,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=(
                    False if "roberta" in self.args.model_checkpoint else True
                ),  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            )

            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(
                -1, self.args.num_neg + 1, max_len
            )
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, self.args.num_neg + 1, max_len
            )
            if "roberta" not in self.args.model_checkpoint:
                p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                    -1, self.args.num_neg + 1, max_len
                )

            # question and passage concat for training
            if "roberta" in self.args.model_checkpoint:
                final_train_dataset = TensorDataset(
                    p_seqs["input_ids"],
                    p_seqs["attention_mask"],
                    q_seqs["input_ids"],
                    q_seqs["attention_mask"],
                )

            else:
                final_train_dataset = TensorDataset(
                    p_seqs["input_ids"],
                    p_seqs["attention_mask"],
                    p_seqs["token_type_ids"],
                    q_seqs["input_ids"],
                    q_seqs["attention_mask"],
                    q_seqs["token_type_ids"],
                )
            with open("final_train_dataset.bin", "wb") as file:
                pickle.dump(final_train_dataset, file)
            return final_train_dataset

    def _load_eval_dataset(self):
        """Set in-batch negative sampling evaluation datasets

        Returns:
            final_train_dataset (TensorDataset): in-batch negative sampling datasets
        """
        if path.isfile("final_valid_dataset.bin"):
            with open("final_valid_dataset.bin", "rb") as file:
                final_valid_dataset = pickle.load(file)
            return final_valid_dataset
        else:
            corpus = np.array(self.eval_dataset["context"])
            cor_titles = np.array(self.eval_dataset["title"])
            titles = []
            p_with_neg = []
            if self.args.neg_strategy == "random":
                for t, c in zip(
                    self.eval_dataset["title"], self.eval_dataset["context"]
                ):
                    titles.append(t)
                    while True:
                        neg_idxs = np.random.randint(
                            len(corpus), size=self.args.num_neg
                        )
                        if not c in corpus[neg_idxs]:
                            p_neg = corpus[neg_idxs]
                            p_neg_title = cor_titles[neg_idxs]
                            p_with_neg.append(c)
                            p_with_neg.extend(p_neg)
                            titles.extend(p_neg_title)
                            break

            elif self.args.neg_strategy == "BM_Gold":
                self.bm25.get_sparse_embedding()
                for t, c, q in zip(
                    self.eval_dataset["title"],
                    self.eval_dataset["context"],
                    self.eval_dataset["question"],
                ):
                    titles.append(t)
                    while True:
                        neg_idxs = np.random.randint(
                            len(corpus), size=self.args.num_neg - 1
                        )
                        if not c in corpus[neg_idxs]:
                            p_neg = corpus[neg_idxs]
                            p_with_neg.append(c)
                            p_neg_title = cor_titles[neg_idxs]
                            # BM25 top1 context 넣기
                            bm_score, bm_context, bm_indice = self.bm25.retrieve(q, 50)
                            i = 0
                            while True:
                                assert i < 50, f"bm25의 top50 까지 {c}와 같아 에러 발생"
                                if bm_context[i] != c:
                                    p_with_neg.append(bm_context[i])
                                    titles.append(self.wiki[str(bm_indice[i])]["title"])
                                    break
                                i += 1
                            p_with_neg.append(bm_context)
                            p_with_neg.extend(p_neg)
                            titles.extend(p_neg_title)
                            break

            q_seqs = self.tokenizer(
                self.eval_dataset["question"],
                padding="max_length",
                truncation=True,
                max_length=100,
                return_tensors="pt",
                return_token_type_ids=(
                    False if "roberta" in self.args.model_checkpoint else True
                ),  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            )
            p_seqs = self.tokenizer(
                titles,
                p_with_neg,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=(
                    False if "roberta" in self.args.model_checkpoint else True
                ),  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            )

            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(
                -1, self.args.num_neg + 1, max_len
            )
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, self.args.num_neg + 1, max_len
            )
            if "roberta" not in self.args.model_checkpoint:
                p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                    -1, self.args.num_neg + 1, max_len
                )

            # question and passage concat for training
            if "roberta" in self.args.model_checkpoint:

                final_valid_dataset = TensorDataset(
                    p_seqs["input_ids"],
                    p_seqs["attention_mask"],
                    q_seqs["input_ids"],
                    q_seqs["attention_mask"],
                )
            else:
                final_valid_dataset = TensorDataset(
                    p_seqs["input_ids"],
                    p_seqs["attention_mask"],
                    p_seqs["token_type_ids"],
                    q_seqs["input_ids"],
                    q_seqs["attention_mask"],
                    q_seqs["token_type_ids"],
                )
            with open("final_valid_dataset.bin", "wb") as file:
                pickle.dump(final_valid_dataset, file)
            return final_valid_dataset
